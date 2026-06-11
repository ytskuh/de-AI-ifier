"""Baseline profiles: corpus -> per-feature percentile bands stored as JSON."""

import json
import re
import statistics
from pathlib import Path

from . import heuristics, lexical, structural

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "profiles"
EXTS = {".md", ".txt", ".tex"}


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix in EXTS)
        elif p.suffix in EXTS:
            files.append(p)
    return files


def _band(values: list[float]) -> list[float]:
    vs = sorted(values)
    q = lambda p: vs[min(len(vs) - 1, max(0, round(p * (len(vs) - 1))))]
    return [round(q(0.05), 3), round(q(0.50), 3), round(q(0.95), 3)]


def build(name: str, paths: list[Path], with_lexical: bool = True) -> dict:
    files = collect_files(paths)
    if len(files) < 3:
        raise SystemExit(f"need >=3 corpus files, got {len(files)}")

    docs, doc_metrics = [], []
    for f in files:
        text = structural.prose(f)
        if len(text.split()) < 300:  # too short for stable per-1k rates
            continue
        docs.append((f.name, text))
        h_findings, metrics = heuristics.run(f)
        if with_lexical:
            lex = lexical.run(f)
            warn = sum(1 for x in lex if x.severity >= 0.6)
            metrics["lexical_warn_per_1k"] = 1000 * warn / max(1, metrics["words"])
            metrics["lexical_all_per_1k"] = 1000 * len(lex) / max(1, metrics["words"])
        doc_metrics.append(metrics)

    feats = structural.biber_features(docs)
    features = {col: _band(feats[col].to_list())
                for col in feats.columns if col != "doc_id"}

    # segment-granularity bands: rate variance shrinks with length, so short
    # targets and per-segment checks need bands from same-sized human text
    seg_docs = []
    for f in files:
        for j, seg in enumerate(structural._segments(f)):
            if seg["words"] >= structural.SEGMENT_WORDS // 2:
                seg_docs.append((f"{f.name}#s{j}", seg["text"]))
    features_seg = {}
    if len(seg_docs) >= 10:
        seg_feats = structural.biber_features(seg_docs)
        features_seg = {col: _band(seg_feats[col].to_list())
                        for col in seg_feats.columns if col != "doc_id"}
    metric_bands = {}
    for key in ("sentence_length_cv", "paragraph_length_cv",
                "lexical_warn_per_1k", "lexical_all_per_1k"):
        vals = [m[key] for m in doc_metrics if key in m]
        if len(vals) >= 3:
            metric_bands[key] = _band(vals)

    return {"name": name, "language": "en", "n_docs": len(docs),
            "docs": [d for d, _ in docs], "features": features,
            "features_seg": features_seg, "n_segments": len(seg_docs),
            "corpus_paths": [str(p) for p in paths],
            "metrics": metric_bands}


def save(profile: dict) -> Path:
    PROFILE_DIR.mkdir(exist_ok=True)
    out = PROFILE_DIR / f"{re.sub(r'[^-a-zA-Z0-9_]', '-', profile['name'])}.json"
    out.write_text(json.dumps(profile, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.json"
    if not path.exists():
        avail = ", ".join(p.stem for p in PROFILE_DIR.glob("*.json")) or "none"
        raise SystemExit(f"no profile '{name}' (available: {avail})")
    return json.loads(path.read_text(encoding="utf-8"))
