"""Statistical layer: observer/performer scoring over configured model pairs.

Design (docs/design/2026-06-10-architecture.md, statistical layer):
- pairs in models/pairs.json; plain-text tokenizer identity asserted per pair
- per token: logprob under both models; delta = lp_perf - lp_obs;
  xent = -sum_v P_obs(v) log P_perf(v)   (ahans30/Binoculars reference formula)
- per sentence/doc on CONTENT tokens only: B = logPPL_perf / xent (low = machine),
  delta quantiles, hot-token clustering, token-class distributions
- efficiency: tokenize once per tokenizer family; evaluate each unique model once
  (cache lp + fp16 log-softmax); derive every pair from the cache. Chunks overlap
  by OVERLAP tokens; only positions with that much context are scored.
- calibration: per-pair human bands (p5/50/95 of doc B and delta) from a human
  corpus, stored in models/stat-bands.json; scores annotated against the band.
- consensus: per-sentence mean normalized B-rank + delta>0 votes across pairs.

Output is always a ranking/annotation, never an AI/human verdict.
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
BANDS_FILE = MODELS / "stat-bands.json"
N_CTX = 4096
OVERLAP = 256
TOP_K = 10           # display cap / uncalibrated-ranking fallback
MIN_UNIT_WORDS = 12   # consecutive prose sentences merge forward into >=12-word units
HOT_WINDOW = 50
_PROBE = "We prove that the sampler converges; results follow."


def load_pairs(only: str | None = None) -> list[dict]:
    cfg = MODELS / "pairs.json"
    if not cfg.exists():
        raise SystemExit(f"no pair registry at {cfg}")
    pairs = [p for p in json.loads(cfg.read_text())
             if (MODELS / p["observer"]).exists() and (MODELS / p["performer"]).exists()]
    if only:
        pairs = [p for p in pairs if p["name"] == only]
        if not pairs:
            raise SystemExit(f"no available pair named '{only}'")
    if not pairs:
        raise SystemExit("no pair has both model files on disk")
    return pairs


def load_bands() -> dict:
    if BANDS_FILE.exists():
        return json.loads(BANDS_FILE.read_text())
    return {}


def _clean(s: str) -> str:
    s = s.replace("⎇", "").replace("**", "").replace("`", "")
    s = re.sub(r"(?<=\s)\*|\*(?=\s)", "", s)
    s = re.sub(r"~+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" ,;:")


def _is_prose(s: str) -> bool:
    """Leaked math/markup is not prose: require half the non-space chars be letters."""
    chars = [ch for ch in s if not ch.isspace()]
    if not chars:
        return False
    letters = sum(ch.isalpha() for ch in chars)
    return letters / len(chars) >= 0.5


def _build_units(sents: list[tuple]) -> list[list[int]]:
    """Merge consecutive prose sentences forward into >=MIN_UNIT_WORDS-word units.
    Every prose sentence lands in exactly one unit; a short trailing remainder
    joins the previous unit. Word-based, so identical across tokenizer families."""
    units, cur, words = [], [], 0
    for i, (_, s) in enumerate(sents):
        if not _is_prose(s):
            continue
        cur.append(i)
        words += len(s.split())
        if words >= MIN_UNIT_WORDS:
            units.append(cur)
            cur, words = [], 0
    if cur:
        if units:
            units[-1] += cur
        else:
            units.append(cur)
    return units


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


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
    from .structural import _nlp
    doc = _nlp()(text)
    byte_of = np.cumsum([0] + [len(c.encode("utf-8")) for c in text])
    starts, ends, labels = [], [], []
    for t in doc:
        if t.is_space:
            continue
        labels.append("transition" if t.text.lower() in TRANSITIONS
                      else _POS_CLASS.get(t.pos_, "symbol"))
        starts.append(int(byte_of[t.idx]))
        ends.append(int(byte_of[t.idx + len(t.text)]))
    starts, ends = np.asarray(starts), np.asarray(ends)
    out = []
    for s, e in tok_span:
        m = (s + e) // 2 if e > s else s
        i = int(np.searchsorted(ends, m, side="right"))
        out.append(labels[i] if i < len(labels) and starts[i] <= m < ends[i] else "symbol")
    return out


class _DocScorer:
    """Scores one document across pairs, evaluating each unique model once."""

    def __init__(self, path: Path):
        self.path = path
        self._family = {}   # family key -> doc tokenization dict
        self._evals = {}    # model file -> (lp per token, [lsm fp16 per chunk])

    def _vocab(self, model_file: str):
        from llama_cpp import Llama
        return Llama(model_path=str(MODELS / model_file), n_ctx=N_CTX,
                     vocab_only=True, verbose=False)

    def _family_key(self, model_file: str) -> str:
        v = self._vocab(model_file)
        key = ",".join(map(str, v.tokenize(_PROBE.encode(), add_bos=False)))
        del v
        return key

    def _tokenize_family(self, key: str, model_file: str) -> dict | None:
        if key in self._family:
            return self._family[key]
        sents = [(line, _clean(s)) for line, p in paragraphs(self.path) for s in sentences(p)]
        sents = [(line, s) for line, s in sents if len(s.split()) >= 4]
        if len(sents) < 5:
            self._family[key] = None
            return None
        text = "\n".join(s for _, s in sents)
        v = self._vocab(model_file)
        tokens = v.tokenize(text.encode("utf-8"), add_bos=True)
        bounds, pos = [], 0
        for _, s in sents:
            pos += len(s.encode("utf-8")) + 1
            bounds.append(pos)
        tok_sent, tok_span, tok_text, cum = [], [], [], 0
        for t in tokens:
            piece = v.detokenize([t])
            tok_span.append((cum, cum + len(piece)))
            tok_text.append(piece.decode("utf-8", errors="replace"))
            cum += len(piece)
            i = int(np.searchsorted(bounds, max(cum - 1, 0), side="right"))
            tok_sent.append(min(i, len(sents) - 1))
        del v
        # overlapping chunks: chunk i starts at i*(N_CTX-OVERLAP); positions with
        # less than OVERLAP context (except in the first chunk) are not scored
        step = N_CTX - OVERLAP
        chunks, scored = [], []
        for s0 in range(0, len(tokens), step):
            chunk = tokens[s0:s0 + N_CTX]
            first_scored = 1 if s0 == 0 else OVERLAP
            chunks.append(chunk)
            scored.append((s0, first_scored, len(chunk)))
            if s0 + N_CTX >= len(tokens):
                break
        self._family[key] = {
            "sents": sents, "tokens": tokens, "tok_sent": np.asarray(tok_sent),
            "classes": np.asarray(_token_classes(text, tok_span)),
            "content": np.array([bool(re.search(r"[a-zA-Z]", t)) for t in tok_text]),
            "chunks": chunks, "scored": scored, "n": len(tokens),
        }
        return self._family[key]

    def _eval(self, model_file: str, fam: dict):
        if model_file in self._evals:
            return self._evals[model_file]
        from llama_cpp import Llama
        llm = Llama(model_path=str(MODELS / model_file), n_ctx=N_CTX,
                    n_gpu_layers=-1, logits_all=True, verbose=False)
        lp = np.full(fam["n"], np.nan, dtype=np.float32)
        lsms = []
        for chunk, (s0, first, clen) in zip(fam["chunks"], fam["scored"]):
            llm.reset()
            llm.eval(chunk)
            logits = np.asarray(llm.scores[:clen], dtype=np.float32)
            lsm = _log_softmax(logits[:-1])
            actual = np.asarray(chunk[1:])
            lp_chunk = lsm[np.arange(clen - 1), actual]
            lp[s0 + first:s0 + clen] = lp_chunk[first - 1:]
            lsms.append(lsm.astype(np.float16))
        del llm
        self._evals[model_file] = (lp, lsms)
        return self._evals[model_file]

    def score(self, pair: dict, bands: dict | None = None) -> dict | None:
        obs, perf = pair["observer"], pair["performer"]
        key_o, key_p = self._family_key(obs), self._family_key(perf)
        if key_o != key_p:
            raise SystemExit(f"pair '{pair['name']}': observer/performer tokenize differently")
        fam = self._tokenize_family(key_o, obs)
        if fam is None:
            return None
        lp_obs, lsm_obs = self._eval(obs, fam)
        lp_perf, lsm_perf = self._eval(perf, fam)

        xent = np.full(fam["n"], np.nan, dtype=np.float32)
        for (s0, first, clen), l_o, l_p in zip(fam["scored"], lsm_obs, lsm_perf):
            p_obs = np.exp(l_o.astype(np.float32))
            xc = -np.einsum("ij,ij->i", p_obs, l_p.astype(np.float32))
            xent[s0 + first:s0 + clen] = xc[first - 1:]

        tok_class = fam["classes"]
        valid = ~np.isnan(lp_obs) & fam["content"]
        delta = lp_perf - lp_obs

        d = delta[valid]
        hot_thresh = float(np.quantile(d, 0.90))
        hot_idx = np.where(valid & (delta > hot_thresh))[0]

        scoreable = ~np.isnan(lp_obs)
        by_class = {}
        for cls in sorted(set(tok_class)):
            m = scoreable & (tok_class == cls)
            if m.sum() < 10:
                continue
            dc = delta[m]
            by_class[cls] = {"n": int(m.sum()),
                             "delta_mean": round(float(dc.mean()), 3),
                             "delta_q90": round(float(np.quantile(dc, 0.9)), 3),
                             "hot_share": round(float((dc > hot_thresh).mean()), 3),
                             "logppl_perf": round(float(-lp_perf[m].mean()), 3)}

        per_sent = []
        for unit in _build_units(fam["sents"]):
            m = np.isin(fam["tok_sent"], unit) & valid
            if m.sum() < 5:
                continue
            line = fam["sents"][unit[0]][0]
            text = " ".join(fam["sents"][i][1] for i in unit)
            per_sent.append({
                "i": unit[0], "line": line, "text": text, "n_sents": len(unit),
                "b": float(-lp_perf[m].mean() / max(xent[m].mean(), 1e-6)),
                "delta": float(delta[m].mean()),
                "hot_share": float((delta[m] > hot_thresh).mean()),
            })

        doc = {
            "b": round(float(-lp_perf[valid].mean() / max(xent[valid].mean(), 1e-6)), 4),
            "delta_mean": round(float(d.mean()), 4),
            "delta_q10": round(float(np.quantile(d, 0.10)), 4),
            "delta_q50": round(float(np.quantile(d, 0.50)), 4),
            "delta_q90": round(float(np.quantile(d, 0.90)), 4),
            "hot_burstiness": round(_burstiness(hot_idx), 3),
            "hot_max_window_share": round(_max_window_share(hot_idx, fam["n"]), 3),
            "content_tokens": int(valid.sum()),
        }
        band = (bands or {}).get(pair["name"])
        if band:
            doc["b_band"] = band["b"]
            doc["delta_band"] = band["delta"]
            doc["b_flag"] = "below human band" if doc["b"] < band["b"][0] else "in band"
            doc["delta_flag"] = ("above human band" if doc["delta_mean"] > band["delta"][2]
                                 else "in band")
        return {"pair": pair["name"], "axis": pair.get("axis", ""),
                "token_classes": by_class, "doc": doc, "sentences": per_sent}


def _burstiness(positions: np.ndarray) -> float:
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
    return _DocScorer(path).score(pair, load_bands())


def run_all(path: Path, only: str | None = None) -> list[dict]:
    scorer = _DocScorer(path)
    bands = load_bands()
    return [r for p in load_pairs(only) if (r := scorer.score(p, bands))]


def consensus(results: list[dict], top: int = TOP_K) -> list[dict]:
    """Cross-pair sentence aggregation: mean normalized B-rank + delta>0 votes."""
    agg = {}
    for r in results:
        ranked = sorted(r["sentences"], key=lambda s: s["b"])
        n = max(1, len(ranked) - 1)
        for rank, s in enumerate(ranked):
            a = agg.setdefault(s["i"], {"line": s["line"], "text": s["text"],
                                        "ranks": [], "bs": [], "votes": 0})
            a["ranks"].append(rank / n)
            a["bs"].append(s["b"])
            a["votes"] += s["delta"] > 0
    rows = []
    for a in agg.values():
        rows.append({"line": a["line"], "text": a["text"],
                     "mean_rank": sum(a["ranks"]) / len(a["ranks"]),
                     "b_min": min(a["bs"]), "delta_votes": a["votes"],
                     "n_pairs": len(a["ranks"])})
    rows.sort(key=lambda r: r["mean_rank"])
    return rows[:top]


def calibrate(paths: list[Path], only_pairs: list[str] | None = None) -> dict:
    """Score a human corpus and store per-pair bands, merging into the existing
    bands file so a newly added pair can be calibrated alone."""
    from .baseline import collect_files
    files = collect_files(paths)
    if len(files) < 3:
        raise SystemExit(f"need >=3 corpus files, got {len(files)}")
    pairs = load_pairs()
    if only_pairs:
        pairs = [p for p in pairs if p["name"] in only_pairs]
    per_pair: dict[str, dict] = {}
    for f in files:
        scorer = _DocScorer(f)
        for p in pairs:
            r = scorer.score(p)
            if r:
                d = per_pair.setdefault(p["name"], {"b": [], "delta": [],
                                                    "sb": [], "sd": []})
                d["b"].append(r["doc"]["b"])
                d["delta"].append(r["doc"]["delta_mean"])
                d["sb"] += [s["b"] for s in r["sentences"]]
                d["sd"] += [s["delta"] for s in r["sentences"]]
        print(f"  calibrated {f.name}")
    q = lambda vs, ps: [round(float(np.quantile(vs, p)), 4) for p in ps]
    bands = load_bands()
    bands.update({name: {"b": q(v["b"], (0.05, 0.50, 0.95)),
                         "delta": q(v["delta"], (0.05, 0.50, 0.95)),
                         "sent_b": q(v["sb"], (0.01, 0.05, 0.50)),
                         "sent_delta": q(v["sd"], (0.50, 0.95, 0.99)),
                         "n_docs": len(v["b"]), "n_sentences": len(v["sb"])}
                  for name, v in per_pair.items()})
    BANDS_FILE.write_text(json.dumps(bands, indent=1) + "\n")
    return bands


def run(path: Path, pair_name: str | None = None,
        corpus_paths: list | None = None) -> tuple[list[Finding], dict]:
    """Findings from the first available (or named) pair — used by report --stat.
    Auto-calibrates the pair against the profile's corpus when bands are missing."""
    import sys
    pair = load_pairs(pair_name)[0]
    if corpus_paths and "sent_b" not in load_bands().get(pair["name"], {}):
        print(f"[stat] pair '{pair['name']}' is uncalibrated — calibrating against the "
              f"profile corpus ({len(corpus_paths)} path(s)); one-time, takes minutes",
              file=sys.stderr)
        calibrate([Path(p) for p in corpus_paths], only_pairs=[pair["name"]])
    res = score_pair(path, pair)
    if res is None:
        return [], {}
    findings, doc = [], res["doc"]
    band = load_bands().get(pair["name"], {})
    band_note = f" ({doc['b_flag']})" if "b_flag" in doc else ""
    if "sent_b" in band:
        b_p01, b_p05, _ = band["sent_b"]
        _, d_p95, d_p99 = band["sent_delta"]
        for s in res["sentences"]:
            if s["b"] < b_p05:
                strong = s["b"] < b_p01
                sev = 0.8 if strong else 0.65
                findings.append(Finding(
                    s["line"], (0, 0), "statistical", f"stat:{pair['name']}:machine_like",
                    sev,
                    f"B={s['b']:.3f} below human sentence band "
                    f"({'p1' if strong else 'p5'}={b_p01 if strong else b_p05:.3f}) on "
                    f"{pair['name']}; doc {doc['b']:.3f}{band_note}. Rewrite in your own "
                    f"words; add a concrete fact or your phrasing.",
                    match=s["text"][:90], payload={"b": s["b"], "delta": s["delta"]}))
            elif s["delta"] > d_p95:
                findings.append(Finding(
                    s["line"], (0, 0), "statistical", f"stat:{pair['name']}:performer_tilt",
                    0.75 if s["delta"] > d_p99 else 0.6,
                    f"Δ={s['delta']:+.3f} above human sentence band (p95={d_p95:+.3f}) — "
                    f"leans toward {pair['axis'] or 'the performer'}.",
                    match=s["text"][:90], payload={"b": s["b"], "delta": s["delta"]}))
    else:  # uncalibrated fallback: ranking only, labeled as such
        for rank, s in enumerate(sorted(res["sentences"], key=lambda x: x["b"])[:TOP_K]):
            findings.append(Finding(
                s["line"], (0, 0), "statistical", f"stat:{pair['name']}:machine_like",
                round(0.7 - 0.02 * rank, 2),
                f"UNCALIBRATED ranking #{rank + 1} on {pair['name']} (B={s['b']:.3f}, "
                f"doc {doc['b']:.3f}) — run `stat --calibrate` for thresholded flags.",
                match=s["text"][:90], payload={"b": s["b"], "delta": s["delta"]}))
    metrics = {f"stat_{k}": v for k, v in doc.items()}
    metrics["stat_pair"] = pair["name"]
    return findings, metrics
