"""Statistical layer: observer/performer scoring over configured model pairs.

Pairs live in models/pairs.json (observer = pretrained base, performer =
instruct/distill with plain-text-identical tokenizer). Per pair, per sentence
(formula matched to ahans30/Binoculars):
- B = logPPL_performer(text) / xent, xent = -sum_v P_observer(v) log P_performer(v).
  LOW B = machine-typical.
- delta = mean logP_performer(tok) - logP_observer(tok): the log-likelihood
  ratio. HIGH = leans toward the performer's (RLHF/vendor) distribution.

Aggregates are computed on CONTENT tokens only (tokens whose text contains a
letter): punctuation, digits, and markup remnants are not authorial choices
and dilute the signal. Beyond means, distribution shape is reported:
- delta quantiles (q10/q50/q90) — a fat right tail betrays localized tilt a
  mean would bury;
- hot-token concentration: hot = delta above the doc's q90; burstiness of
  gaps between hot tokens (-1 regular .. 0 Poisson .. +1 clustered) and the
  max share of hot tokens in any 50-token window. Clustered hot tokens =
  specific machine-flavored passages; spread = global register effect.

Output is a RANKING, never a verdict — thresholds don't transfer across pairs.
Requires the [stat] extra (llama-cpp-python).
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from .findings import Finding
from .heuristics import paragraphs, sentences

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("DEAIIFY_MODELS", ROOT / "models"))
N_CTX = 4096
TOP_K = 10
HOT_WINDOW = 50


def load_pairs(only: str | None = None) -> list[dict]:
    cfg = MODELS / "pairs.json"
    if not cfg.exists():
        raise SystemExit(f"no pair registry at {cfg}")
    pairs = json.loads(cfg.read_text())
    pairs = [p for p in pairs if (MODELS / p["observer"]).exists()
             and (MODELS / p["performer"]).exists()]
    if only:
        pairs = [p for p in pairs if p["name"] == only]
        if not pairs:
            raise SystemExit(f"no available pair named '{only}'")
    if not pairs:
        raise SystemExit("no pair has both model files on disk")
    return pairs


def _clean(s: str) -> str:
    s = s.replace("⎇", "").replace("**", "").replace("`", "")
    s = re.sub(r"(?<=\s)\*|\*(?=\s)", "", s)
    s = re.sub(r"~+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" ,;:")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _doc_tokens(path: Path, model_path: Path):
    """Sentence inventory + tokenization + per-token sentence index, text, byte span."""
    from llama_cpp import Llama

    sents = [(line, _clean(s)) for line, p in paragraphs(path) for s in sentences(p)]
    sents = [(line, s) for line, s in sents if len(s.split()) >= 4]
    if len(sents) < 5:
        return None
    text = "\n".join(s for _, s in sents)
    llm = Llama(model_path=str(model_path), n_ctx=N_CTX, vocab_only=True, verbose=False)
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
    bounds, pos = [], 0
    for _, s in sents:
        pos += len(s.encode("utf-8")) + 1
        bounds.append(pos)
    tok_sent, tok_text, tok_span, cum = [], [], [], 0
    for t in tokens:
        piece = llm.detokenize([t])
        tok_span.append((cum, cum + len(piece)))
        cum += len(piece)
        i = int(np.searchsorted(bounds, max(cum - 1, 0), side="right"))
        tok_sent.append(min(i, len(sents) - 1))
        tok_text.append(piece.decode("utf-8", errors="replace"))
    del llm
    return sents, text, tokens, np.asarray(tok_sent), tok_text, tok_span


TRANSITIONS = {"however", "moreover", "furthermore", "additionally", "thus", "therefore",
               "hence", "consequently", "nevertheless", "nonetheless", "meanwhile",
               "specifically", "notably", "importantly", "interestingly", "overall",
               "finally", "indeed", "instead", "accordingly", "similarly", "conversely",
               "first", "second", "third", "ultimately", "crucially"}
_POS_CLASS = {"PROPN": "proper_noun", "NOUN": "noun", "VERB": "verb", "AUX": "verb",
              "ADJ": "adjective", "ADV": "adverb", "NUM": "number", "PUNCT": "punctuation",
              "PRON": "function", "DET": "function", "ADP": "function", "CCONJ": "function",
              "SCONJ": "function", "PART": "function"}


def _token_classes(text: str, tok_span: list[tuple]) -> list[str]:
    """Class label per model token, via spaCy POS aligned in byte space."""
    from .structural import _nlp
    doc = _nlp()(text)
    byte_of = np.cumsum([0] + [len(c.encode("utf-8")) for c in text])
    starts, ends, labels = [], [], []
    for t in doc:
        if t.is_space:
            continue
        label = ("transition" if t.text.lower() in TRANSITIONS
                 else _POS_CLASS.get(t.pos_, "symbol"))
        starts.append(int(byte_of[t.idx]))
        ends.append(int(byte_of[t.idx + len(t.text)]))
        labels.append(label)
    starts, ends = np.asarray(starts), np.asarray(ends)
    out = []
    for s, e in tok_span:
        m = (s + e) // 2 if e > s else s
        i = int(np.searchsorted(ends, m, side="right"))
        out.append(labels[i] if i < len(labels) and starts[i] <= m < ends[i] else "symbol")
    return out


def _score(model_path: Path, chunks: list[list[int]]):
    """Per-token logprob of actual tokens + per-chunk fp16 log-softmax."""
    from llama_cpp import Llama

    llm = Llama(model_path=str(model_path), n_ctx=N_CTX, n_gpu_layers=-1,
                logits_all=True, verbose=False)
    lps, lsms = [], []
    for chunk in chunks:
        llm.reset()
        llm.eval(chunk)
        logits = np.asarray(llm.scores[:len(chunk)], dtype=np.float32)
        lsm = _log_softmax(logits[:-1])
        actual = np.asarray(chunk[1:])
        lp = np.full(len(chunk), np.nan, dtype=np.float32)
        lp[1:] = lsm[np.arange(len(actual)), actual]
        lps.append(lp)
        lsms.append(lsm.astype(np.float16))
    del llm
    return np.concatenate(lps), lsms


def _burstiness(positions: np.ndarray, n: int) -> float:
    """Gap burstiness of hot-token positions: -1 regular, 0 Poisson, +1 clustered."""
    if len(positions) < 3:
        return 0.0
    gaps = np.diff(np.sort(positions))
    mu, sd = gaps.mean(), gaps.std()
    return float((sd - mu) / (sd + mu)) if (sd + mu) > 0 else 0.0


def _max_window_share(positions: np.ndarray, n: int, w: int = HOT_WINDOW) -> float:
    if len(positions) == 0 or n <= w:
        return 0.0
    marks = np.zeros(n)
    marks[positions] = 1
    csum = np.cumsum(marks)
    return float((csum[w:] - csum[:-w]).max() / w)


def score_pair(path: Path, pair: dict) -> dict | None:
    obs, perf = MODELS / pair["observer"], MODELS / pair["performer"]
    doc = _doc_tokens(path, obs)
    if doc is None:
        return None
    sents, text, tokens, tok_sent, tok_text, tok_span = doc
    tok_class = np.asarray(_token_classes(text, tok_span))

    # plain-text tokenizer equality assertion (special tokens may differ)
    from llama_cpp import Llama
    probe = "We prove that the sampler converges; results follow."
    t1 = Llama(model_path=str(obs), vocab_only=True, verbose=False).tokenize(probe.encode(), add_bos=False)
    t2 = Llama(model_path=str(perf), vocab_only=True, verbose=False).tokenize(probe.encode(), add_bos=False)
    if t1 != t2:
        raise SystemExit(f"pair '{pair['name']}': observer/performer tokenize differently")

    chunks = [tokens[i:i + N_CTX] for i in range(0, len(tokens), N_CTX)]
    lp_obs, lsm_obs = _score(obs, chunks)
    lp_perf_parts, xent_parts = [], []
    from llama_cpp import Llama as _L
    llm = _L(model_path=str(perf), n_ctx=N_CTX, n_gpu_layers=-1, logits_all=True, verbose=False)
    for chunk, lsm1 in zip(chunks, lsm_obs):
        llm.reset()
        llm.eval(chunk)
        logits = np.asarray(llm.scores[:len(chunk)], dtype=np.float32)[:-1]
        lsm2 = _log_softmax(logits)
        actual = np.asarray(chunk[1:])
        lp = np.full(len(chunk), np.nan, dtype=np.float32)
        lp[1:] = lsm2[np.arange(len(actual)), actual]
        lp_perf_parts.append(lp)
        p_obs = np.exp(lsm1.astype(np.float32))
        xent_parts.append(np.concatenate([[np.nan], -np.einsum("ij,ij->i", p_obs, lsm2)]))
    del llm
    lp_perf = np.concatenate(lp_perf_parts)
    xent = np.concatenate(xent_parts)

    content = np.array([bool(re.search(r"[a-zA-Z]", t)) for t in tok_text])
    valid = ~np.isnan(lp_obs) & content
    delta = lp_perf - lp_obs

    d = delta[valid]
    hot_thresh = float(np.quantile(d, 0.90))
    hot_idx = np.where(valid & (delta > hot_thresh))[0]
    n = int(valid.sum())

    per_sent = []
    for i, (line, s) in enumerate(sents):
        m = (tok_sent == i) & valid
        all_m = (tok_sent == i) & ~np.isnan(lp_obs)
        # mostly-symbol "sentences" are leaked math/markup, not prose
        if m.sum() < 5 or m.sum() < 0.6 * all_m.sum():
            continue
        per_sent.append({
            "i": i, "line": line, "text": s,
            "b": float(-lp_perf[m].mean() / max(xent[m].mean(), 1e-6)),
            "delta": float(delta[m].mean()),
            "delta_q90": float(np.quantile(delta[m], 0.9)),
            "hot_share": float((delta[m] > hot_thresh).mean()),
        })

    # score distribution by token class (all scoreable tokens, incl. punctuation)
    scoreable = ~np.isnan(lp_obs)
    by_class = {}
    for cls in sorted(set(tok_class)):
        m = scoreable & (tok_class == cls)
        if m.sum() < 10:
            continue
        dc = delta[m]
        by_class[cls] = {
            "n": int(m.sum()),
            "delta_mean": round(float(dc.mean()), 3),
            "delta_q90": round(float(np.quantile(dc, 0.9)), 3),
            "hot_share": round(float((dc > hot_thresh).mean()), 3),
            "logppl_perf": round(float(-lp_perf[m].mean()), 3),
        }

    return {
        "pair": pair["name"], "axis": pair.get("axis", ""),
        "token_classes": by_class,
        "doc": {
            "b": round(float(-lp_perf[valid].mean() / max(xent[valid].mean(), 1e-6)), 4),
            "delta_mean": round(float(d.mean()), 4),
            "delta_q10": round(float(np.quantile(d, 0.10)), 4),
            "delta_q50": round(float(np.quantile(d, 0.50)), 4),
            "delta_q90": round(float(np.quantile(d, 0.90)), 4),
            "hot_burstiness": round(_burstiness(hot_idx, n), 3),
            "hot_max_window_share": round(_max_window_share(hot_idx, len(tokens)), 3),
            "content_tokens": n,
        },
        "sentences": per_sent,
    }


def run(path: Path, pair_name: str | None = None) -> tuple[list[Finding], dict]:
    """Findings from the first available (or named) pair — used by report --stat."""
    pair = load_pairs(pair_name)[0]
    res = score_pair(path, pair)
    if res is None:
        return [], {}
    findings, doc = [], res["doc"]
    for rank, s in enumerate(sorted(res["sentences"], key=lambda x: x["b"])[:TOP_K]):
        findings.append(Finding(
            s["line"], (0, 0), "statistical", f"stat:{pair['name']}:machine_like",
            round(0.7 - 0.02 * rank, 2),
            f"Most machine-like #{rank + 1} on {pair['name']} (B={s['b']:.3f}, doc {doc['b']:.3f}): "
            f"rewrite in your own words; add a concrete fact or your phrasing.",
            match=s["text"][:90], payload={"b": s["b"], "delta": s["delta"]}))
    for rank, s in enumerate(sorted(res["sentences"], key=lambda x: -x["delta"])[:TOP_K]):
        findings.append(Finding(
            s["line"], (0, 0), "statistical", f"stat:{pair['name']}:performer_tilt",
            round(0.6 - 0.02 * rank, 2),
            f"Most {pair['axis'] or 'performer'}-tilted #{rank + 1} "
            f"(Δ={s['delta']:+.3f}, doc {doc['delta_mean']:+.3f}).",
            match=s["text"][:90], payload={"b": s["b"], "delta": s["delta"]}))
    metrics = {f"stat_{k}": v for k, v in doc.items()}
    metrics["stat_pair"] = pair["name"]
    return findings, metrics


def run_all(path: Path) -> list[dict]:
    return [r for p in load_pairs() if (r := score_pair(path, p))]
