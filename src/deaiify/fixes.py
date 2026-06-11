"""Mechanical fix engine: deterministic, meaning-preserving edits as a unified diff.

Three fix classes, all reviewable as a diff before --write:
- replace: substitution rules carrying exactly one single-word replacement
- delete:  filler words/phrases that add no information (additive transitions,
           hedging lead-ins) — never connectives that carry logic (However, Thus)
- split:   semicolon splice -> two sentences

Everything else (tricolons, gerund recasting, voice) needs judgment and stays
in the report for the assisted/human passes.
"""

import difflib
import re
from pathlib import Path

from .findings import Finding

# rules whose matches are safe to delete outright
DELETE_RULES = {"Deaiify.NousTier3Filler", "ai-tells.HedgingPhrases"}
# transition words that are purely additive — deletion preserves the argument.
# Contrastive/causal ones (However, Thus, Therefore, Nevertheless) are kept.
DELETABLE_TRANSITIONS = {"specifically", "moreover", "furthermore", "additionally",
                         "notably", "importantly", "interestingly", "thereby"}
SPLIT_RULES = {"ai-tells.SemicolonUsage"}


def _classify(f: Finding) -> str | None:
    act = f.payload.get("action") or {}
    params = act.get("Params") or []
    if (act.get("Name") == "replace" and len(params) == 1
            and " " not in f.match.strip() and " " not in params[0].strip()):
        return "replace"
    if f.rule in DELETE_RULES:
        return "delete"
    if (f.rule == "ai-tells.FormalTransitions"
            and f.match.strip().lower().rstrip(",") in DELETABLE_TRANSITIONS):
        return "delete"
    if f.rule in SPLIT_RULES:
        return "split"
    return None


def fixable(findings: list[Finding]) -> list[tuple[Finding, str]]:
    return [(f, kind) for f in findings
            if f.line > 0 and (kind := _classify(f)) is not None]


def _edit(line: str, f: Finding, kind: str) -> tuple[int, int, str] | None:
    """Return (start0, end0, replacement) as 0-based slice, or None to skip."""
    s, e = f.span[0] - 1, f.span[1]  # vale spans: 1-based inclusive
    if line[s:e] != f.match:
        return None
    if kind == "replace":
        repl = f.payload["action"]["Params"][0]
        if f.match[:1].isupper() and repl[:1].islower():
            repl = repl[:1].upper() + repl[1:]
        return s, e, repl

    if kind == "split":
        m = re.match(r";\s*", line[s:e] if ";" in line[s:e] else line[s:])
        if not m or s + len(m.group(0)) >= len(line):
            return None
        nxt = line[s + len(m.group(0))]
        if not nxt.isalpha():  # don't split inside math/markup lists
            return None
        return s, s + len(m.group(0)) + 1, ". " + nxt.upper()

    # delete: swallow trailing comma/space, recapitalize if sentence-initial
    end = e
    m = re.match(r",?\s*", line[end:])
    end += m.end() if m else 0
    sentence_initial = s == 0 or re.search(r"[.!?]\s+$", line[:s])
    repl = ""
    if sentence_initial and end < len(line) and line[end].islower():
        return s, end + 1, line[end].upper()
    return s, end, repl


def apply(path: Path, findings: list[Finding], write: bool = False) -> tuple[str, int]:
    """Return (unified diff, n_applied); write the result if write=True."""
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = list(original)
    by_line: dict[int, list[tuple[Finding, str]]] = {}
    for f, kind in fixable(findings):
        by_line.setdefault(f.line, []).append((f, kind))

    applied = 0
    for ln, fs in by_line.items():
        if ln > len(lines):
            continue
        line = lines[ln - 1]
        fs.sort(key=lambda t: -t[0].span[0])  # right-to-left keeps spans valid
        last_start = len(line) + 1
        for f, kind in fs:
            ed = _edit(line, f, kind)
            if ed is None or ed[1] > last_start:
                continue
            s0, e0, repl = ed
            line = line[:s0] + repl + line[e0:]
            last_start = s0
            applied += 1
        lines[ln - 1] = re.sub(r"  +", " ", line) if line.strip() else line

    diff = "".join(difflib.unified_diff(original, lines, str(path), f"{path} (deaiify fix)"))
    if write and applied:
        path.write_text("".join(lines), encoding="utf-8")
    return diff, applied
