"""deaiify CLI — report / fix / check / baseline / stat."""

import argparse
import sys
from pathlib import Path

from . import findings as F
from . import heuristics, lexical

ROOT = Path(__file__).resolve().parents[2]


def _setup_hints() -> None:
    """One-line nudges for a fresh checkout; silent once everything exists."""
    hints = []
    if not (ROOT / "bin" / "vale").exists():
        hints.append("lexical layer needs the Vale binary in bin/ — see README §Setup")
    if not list((ROOT / "models").glob("*.gguf")):
        hints.append("statistical layer needs GGUF detector models in models/ — "
                     "see README §Models (pairs are configured in models/pairs.json)")
    if not (ROOT / "data" / "baseline").exists():
        hints.append("no baseline corpora in data/ — add your own pre-LLM writing and/or "
                     "sampled arXiv papers (tools/collect_topical_baseline.py), then run "
                     "`deaiify baseline build` — see README §Data")
    for h in hints:
        print(f"[setup] {h}", file=sys.stderr)


def _all_findings(path: Path, profile_name: str | None, with_stat: bool = False):
    lex = lexical.run(path)
    heur, metrics = heuristics.run(path)
    extra = []
    if profile_name:
        from . import baseline, structural
        extra = structural.run(path, baseline.load(profile_name))
    if with_stat:
        from . import statistical
        sfind, smetrics = statistical.run(path)
        extra += sfind
        metrics.update(smetrics)
    return F.merge(lex + heur + extra), metrics


def cmd_report(args) -> int:
    merged, metrics = _all_findings(args.file, args.profile, getattr(args, "stat", False))
    merged = [f for f in merged if f.severity >= args.min_severity]
    words = max(1, metrics.get("words", 1))
    meta = {"file": str(args.file), **metrics,
            "findings_per_1k": 1000 * len(merged) / words}
    if args.json:
        print(F.to_json(merged, meta))
    else:
        F.render_terminal(merged, meta)
    return 0


def cmd_fix(args) -> int:
    from . import fixes
    lex = lexical.run(args.file)
    diff, n = fixes.apply(args.file, lex, write=args.write)
    if not n:
        print("no mechanically-fixable findings")
        return 0
    print(diff)
    print(f"{n} fixes {'applied' if args.write else 'available (rerun with --write to apply)'}")
    return 0


def cmd_check(args) -> int:
    from . import baseline
    prof = baseline.load(args.profile)
    merged, metrics = _all_findings(args.file, args.profile)
    words = max(1, metrics.get("words", 1))
    lex_warn = sum(1 for f in merged if f.layer == "lexical" and f.severity >= 0.6)
    metrics["lexical_warn_per_1k"] = 1000 * lex_warn / words
    structural_out = [f for f in merged if f.layer == "structural"]

    failures = []
    for key, (p05, p50, p95) in prof.get("metrics", {}).items():
        v = metrics.get(key)
        if v is None:
            continue
        ok = v <= p95 * 1.1 if "per_1k" in key else v >= p05 * 0.9
        status = "ok" if ok else "FAIL"
        print(f"  {status:4s} {key}: {v:.2f}  (baseline band [{p05:.2f}–{p95:.2f}])")
        if not ok:
            failures.append(key)
    key_tells = [f for f in structural_out if f.payload.get("key_tell")]
    status = "ok" if len(key_tells) == 0 else "FAIL"
    print(f"  {status:4s} structural key-tell features out of band: {len(key_tells)} "
          f"({len(structural_out)} total out of band)")
    for f in structural_out:
        print(f"       - {f.message}")
    if key_tells:
        failures.append("structural_key_tells")

    print("PASS" if not failures else f"FAIL ({', '.join(failures)})")
    return 0 if not failures else 1


def cmd_stat(args) -> int:
    from . import statistical
    if args.pair and args.pair != "all":
        results = [statistical.score_pair(args.file, statistical.load_pairs(args.pair)[0])]
    else:
        results = statistical.run_all(args.file)
    for r in results:
        if not r:
            continue
        d = r["doc"]
        print(f"[{r['pair']}] {r['axis']}")
        print(f"  B={d['b']} (low=machine)  Δ mean={d['delta_mean']:+.3f} "
              f"q10/50/90={d['delta_q10']:+.3f}/{d['delta_q50']:+.3f}/{d['delta_q90']:+.3f}")
        print(f"  hot-token burstiness={d['hot_burstiness']:+.2f} (0=random, +1=clustered)  "
              f"max {statistical.HOT_WINDOW}-tok window share={d['hot_max_window_share']:.2f}  "
              f"content tokens={d['content_tokens']}")
        if args.classes and r.get("token_classes"):
            print(f"  {'class':12s} {'n':>5s} {'Δmean':>7s} {'Δq90':>7s} {'hot%':>5s} {'logPPL':>7s}")
            for cls, c in sorted(r["token_classes"].items(), key=lambda kv: -kv[1]["delta_mean"]):
                print(f"  {cls:12s} {c['n']:5d} {c['delta_mean']:+7.3f} {c['delta_q90']:+7.3f} "
                      f"{100*c['hot_share']:4.0f}% {c['logppl_perf']:7.3f}")
        for s in sorted(r["sentences"], key=lambda x: x["b"])[:args.top]:
            print(f"    L{s['line']:4d} B={s['b']:.3f} Δ={s['delta']:+.3f} hot={s['hot_share']:.2f}  {s['text'][:64]!r}")
    return 0


def cmd_baseline_build(args) -> int:
    from . import baseline
    prof = baseline.build(args.name, args.paths)
    out = baseline.save(prof)
    print(f"profile '{prof['name']}' from {prof['n_docs']} docs -> {out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="deaiify",
                                 description="Localize and fix AI-writing tells in an article.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="rank AI-tell hotspots in a file")
    rep.add_argument("file", type=Path)
    rep.add_argument("--json", action="store_true")
    rep.add_argument("--min-severity", type=float, default=0.0)
    rep.add_argument("--profile", help="baseline profile for structural comparison")
    rep.add_argument("--stat", action="store_true",
                     help="add statistical layer (needs [stat] extra + GGUF pair in models/)")
    rep.set_defaults(fn=cmd_report)

    st = sub.add_parser("stat", help="statistical scores (B + performer-tilt, per model pair)")
    st.add_argument("file", type=Path)
    st.add_argument("--pair", default="all", help="pair name from models/pairs.json (default: all)")
    st.add_argument("--top", type=int, default=3, help="top machine-like sentences per pair")
    st.add_argument("--classes", action="store_true",
                    help="show score distribution by token class (POS, transitions, punctuation)")
    st.set_defaults(fn=cmd_stat)

    fix = sub.add_parser("fix", help="apply unambiguous substitutions (diff preview)")
    fix.add_argument("file", type=Path)
    fix.add_argument("--write", action="store_true", help="write changes to the file")
    fix.set_defaults(fn=cmd_fix)

    chk = sub.add_parser("check", help="pass/fail vs a baseline profile")
    chk.add_argument("file", type=Path)
    chk.add_argument("--profile", required=True)
    chk.set_defaults(fn=cmd_check)

    bl = sub.add_parser("baseline", help="manage baseline profiles")
    bls = bl.add_subparsers(dest="bcmd", required=True)
    blb = bls.add_parser("build", help="build a profile from corpus files/dirs")
    blb.add_argument("--name", required=True)
    blb.add_argument("paths", nargs="+", type=Path)
    blb.set_defaults(fn=cmd_baseline_build)

    args = ap.parse_args()
    _setup_hints()
    if hasattr(args, "file") and not args.file.exists():
        sys.exit(f"no such file: {args.file}")
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
