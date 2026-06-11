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

    if doc_level:
        con.print("[bold]Document-level[/] (whole-text rates and scores, not occurrence counts):")
        for f in sorted(doc_level, key=lambda f: -f.severity):
            con.print(f"  • {f.message}", style=sev(f.severity))
        con.print()

    by_rule: dict[str, list[Finding]] = {}
    for f in line_level:
        by_rule.setdefault(f.rule, []).append(f)
    top = Table(title="Top rules (occurrence counts)", show_lines=False)
    top.add_column("hits", justify="right")
    top.add_column("rule")
    top.add_column("example", overflow="fold", max_width=50)
    for rule, fs in sorted(by_rule.items(), key=lambda kv: -len(kv[1]))[:15]:
        top.add_row(str(len(fs)), rule, fs[0].match or fs[0].message[:50],
                    style=sev(max(f.severity for f in fs)))
    con.print(top)

    tbl = Table(title="Findings by location")
    tbl.add_column("line", justify="right")
    tbl.add_column("sev")
    tbl.add_column("match", max_width=28, overflow="fold")
    tbl.add_column("message", overflow="fold")
    for f in line_level:
        label = "ERR" if f.severity >= 0.9 else ("WARN" if f.severity >= 0.6 else "sugg")
        tbl.add_row(str(f.line), label, f.match, f.message, style=sev(f.severity))
    con.print(tbl)
