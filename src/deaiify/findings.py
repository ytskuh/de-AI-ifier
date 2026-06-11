"""Shared finding model, overlap merge, and report rendering."""

import json
from dataclasses import asdict, dataclass, field

SEVERITY_WEIGHT = {"suggestion": 0.3, "warning": 0.6, "error": 0.9}


@dataclass
class Finding:
    line: int            # 1-based; 0 = document-level
    span: tuple          # (start_col, end_col), 1-based; (0, 0) = whole line/doc
    layer: str           # lexical | uniformity | genericity
    rule: str            # e.g. "ai-tells.ServesAsDodge"
    severity: float      # 0-1
    message: str
    match: str = ""
    payload: dict = field(default_factory=dict)


def merge(findings: list[Finding]) -> list[Finding]:
    """Drop findings whose span overlaps a higher-severity finding on the same line."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: -f.severity):
        clash = any(
            k.line == f.line and f.line > 0 and k.span[0] < f.span[1] and f.span[0] < k.span[1]
            for k in kept
        )
        if not clash:
            kept.append(f)
    return sorted(kept, key=lambda f: (f.line, f.span[0]))


def to_json(findings: list[Finding], meta: dict) -> str:
    return json.dumps({"meta": meta, "findings": [asdict(f) for f in findings]},
                      indent=2, ensure_ascii=False)


def render_terminal(findings: list[Finding], meta: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()
    sev = lambda s: "red" if s >= 0.9 else ("yellow" if s >= 0.6 else "dim")

    con.print(f"\n[bold]{meta['file']}[/] — {meta['words']} words, "
              f"{len(findings)} findings ({meta['findings_per_1k']:.0f}/1k words)\n")

    doc_level = [f for f in findings if f.line == 0]
    line_level = [f for f in findings if f.line > 0]

    biber = sorted((f for f in doc_level if "gloss" in f.payload),
                   key=lambda f: -f.payload["excess"])
    seg = sorted((f for f in doc_level if "gloss_seg" in f.payload),
                 key=lambda f: f.payload["p"])
    other_doc = [f for f in doc_level
                 if "gloss" not in f.payload and "gloss_seg" not in f.payload]

    con.print("[bold]── Document profile ──[/]")
    if biber:
        basis = biber[0].payload.get("band_basis", "doc")
        tbl = Table(title=f"Structural rates vs human band — {basis} "
                          f"(↑ above / ↓ below — both directions matter; sorted by σ)")
        tbl.add_column("feature", overflow="fold", max_width=34)
        tbl.add_column("rate/1k", justify="right")
        tbl.add_column("", justify="center")  # direction arrow
        tbl.add_column("band (p5–p95)", justify="right")
        tbl.add_column("σ", justify="right")
        tbl.add_column("edit hint", overflow="fold")
        for f in biber:
            p = f.payload
            p05, p50, p95 = p["band"]
            tbl.add_row(p["gloss"], f"{p['value']:.1f}", p["arrow"],
                        f"{p05:.1f}–{p95:.1f}", f"{p['excess']:.1f}×",
                        p["hint"] or "—", style=sev(f.severity))
        con.print(tbl)
    if seg:
        n_seg = len(seg[0].payload["map"])
        st = Table(title=f"Segments vs HUMAN segment band ({n_seg} segments; map: "
                         f"↑ above / · in band / ↓ below the profile's 1k-segment band)")
        st.add_column("feature", overflow="fold", max_width=30)
        st.add_column("p", justify="right")
        st.add_column("rate/1k", justify="right")
        st.add_column("band?", justify="center")
        st.add_column("segment map")
        st.add_column("worst segments", overflow="fold")
        for f in seg:
            p = f.payload
            p_str = f"≤{p['p']:.4f}" if p.get("at_floor") else f"{p['p']:.4f}"
            st.add_row(p["gloss_seg"], p_str, f"{p['doc_rate']:.1f}",
                       "in" if p["in_band"] else "out", p["map"],
                       "; ".join(p["outliers"]), style=sev(f.severity))
        con.print(st)
    for f in sorted(other_doc, key=lambda f: -f.severity):
        con.print(f"  • {f.message}", style=sev(f.severity))
    if biber or seg or other_doc:
        con.print()

    by_rule: dict[str, list[Finding]] = {}
    for f in line_level:
        by_rule.setdefault(f.rule, []).append(f)
    recurring = [f"{rule}×{len(fs)}" for rule, fs in
                 sorted(by_rule.items(), key=lambda kv: -len(kv[1])) if len(fs) >= 3]
    if recurring:
        con.print("[bold]── Recurring patterns ──[/]  " + "  ".join(recurring[:10]) + "\n")

    tbl = Table(title="Edit worklist (top to bottom alongside the document)")
    tbl.add_column("line", justify="right")
    tbl.add_column("sev")
    tbl.add_column("match", max_width=28, overflow="fold")
    tbl.add_column("what to do", overflow="fold")
    for f in line_level:
        label = "ERR" if f.severity >= 0.9 else ("WARN" if f.severity >= 0.6 else "sugg")
        tbl.add_row(str(f.line), label, f.match, f.message, style=sev(f.severity))
    con.print(tbl)
