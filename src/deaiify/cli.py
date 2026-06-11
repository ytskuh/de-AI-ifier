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
        corpus = None
        if profile_name:
            from . import baseline
            corpus = baseline.load(profile_name).get("corpus_paths")
        sfind, smetrics = statistical.run(path, corpus_paths=corpus)
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
    print("deaiify fix is DISABLED: the mechanical fix engine produced unsafe edits "
          "(multi-word collisions, part-of-speech swaps, comparative-context swaps) and is "
          "banned until span-safety work lands — see the design doc. Use `deaiify report` "
          "and edit the flagged spans yourself.", file=sys.stderr)
    return 2


def cmd_stat(args) -> int:
    from . import statistical
    if args.calibrate:
        bands = statistical.calibrate(args.calibrate)
        for name, b in bands.items():
            print(f"  {name}: B band {b['b']}  Δ band {b['delta']}  (n={b['n_docs']})")
        print(f"bands -> {statistical.BANDS_FILE}")
        return 0
    if args.file is None:
        sys.exit("stat needs a file (or --calibrate <corpus>)")
    results = statistical.run_all(args.file, None if args.pair == "all" else args.pair)
    if args.consensus:
        rows = statistical.consensus(results, top=args.top or 10)
        print(f"consensus over {len(results)} pairs (low rank = machine-like on many axes; "
              f"Δ votes = pairs leaning toward their vendor):")
        for r in rows:
            print(f"  L{r['line']:4d} rank={r['mean_rank']:.2f} Bmin={r['b_min']:.2f} "
                  f"Δ>0 on {r['delta_votes']}/{r['n_pairs']}  {r['text'][:60]!r}")
        return 0
    for r in results:
        if not r:
            continue
        d = r["doc"]
        print(f"[{r['pair']}] {r['axis']}")
        band = f"  [human band {d['b_band'][0]}–{d['b_band'][2]}: {d['b_flag']}]" if "b_band" in d else ""
        print(f"  B={d['b']} (low=machine){band}  Δ mean={d['delta_mean']:+.3f} "
              f"q10/50/90={d['delta_q10']:+.3f}/{d['delta_q50']:+.3f}/{d['delta_q90']:+.3f}")
        print(f"  hot-token burstiness={d['hot_burstiness']:+.2f} (0=random, +1=clustered)  "
              f"max {statistical.HOT_WINDOW}-tok window share={d['hot_max_window_share']:.2f}  "
              f"content tokens={d['content_tokens']}")
        if args.classes and r.get("token_classes"):
            print(f"  {'class':12s} {'n':>5s} {'Δmean':>7s} {'Δq90':>7s} {'hot%':>5s} {'logPPL':>7s}")
            for cls, c in sorted(r["token_classes"].items(), key=lambda kv: -kv[1]["delta_mean"]):
                print(f"  {cls:12s} {c['n']:5d} {c['delta_mean']:+7.3f} {c['delta_q90']:+7.3f} "
                      f"{100*c['hot_share']:4.0f}% {c['logppl_perf']:7.3f}")
        band = statistical.load_bands().get(r["pair"], {})
        if "sent_b" in band:
            b_p05 = band["sent_b"][1]
            d_p95 = band["sent_delta"][1]
            flagged = [s for s in r["sentences"] if s["b"] < b_p05 or s["delta"] > d_p95]
            flagged.sort(key=lambda s: s["b"])
            share = 100 * len(flagged) / max(1, len(r["sentences"]))
            print(f"  flagged units (B<p5={b_p05:.3f} or Δ>p95={d_p95:+.3f}): "
                  f"{len(flagged)} of {len(r['sentences'])} = {share:.0f}% "
                  f"(chance level ≈10% — human text flags that much by construction)")
            for s in flagged[:args.top or len(flagged)]:
                print(f"    L{s['line']:4d} B={s['b']:.3f} Δ={s['delta']:+.3f}  {s['text'][:64]!r}")
            if args.top and len(flagged) > args.top:
                print(f"    … +{len(flagged) - args.top} more (raise --top)")
        else:
            for s in sorted(r["sentences"], key=lambda x: x["b"])[:args.top]:
                print(f"    L{s['line']:4d} B={s['b']:.3f} Δ={s['delta']:+.3f} hot={s['hot_share']:.2f}  {s['text'][:64]!r} [uncalibrated ranking]")
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
    st.add_argument("file", type=Path, nargs="?")
    st.add_argument("--pair", default="all", help="pair name from models/pairs.json (default: all)")
    st.add_argument("--top", type=int, default=3, help="top machine-like sentences per pair")
    st.add_argument("--classes", action="store_true",
                    help="show score distribution by token class (POS, transitions, punctuation)")
    st.add_argument("--consensus", action="store_true",
                    help="aggregate sentence rankings across all pairs")
    st.add_argument("--calibrate", nargs="+", type=Path, metavar="CORPUS",
                    help="score a human corpus on every pair and store per-pair bands")
    st.set_defaults(fn=cmd_stat)

    fix = sub.add_parser("fix", help="DISABLED pending precision work (see design doc)")
    fix.add_argument("file", type=Path)
    fix.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    fix.set_defaults(fn=cmd_fix)

    bl = sub.add_parser("baseline", help="manage baseline profiles")
    bls = bl.add_subparsers(dest="bcmd", required=True)
    blb = bls.add_parser("build", help="build a profile from corpus files/dirs")
    blb.add_argument("--name", required=True)
    blb.add_argument("paths", nargs="+", type=Path)
    blb.set_defaults(fn=cmd_baseline_build)

    args = ap.parse_args()
    _setup_hints()
    if getattr(args, "file", None) is not None and not args.file.exists():
        sys.exit(f"no such file: {args.file}")
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
