"""Heuristic layer: document-uniformity and content-genericity signals.

Pure regex/statistics — no models. Works on markdown/plain text and on LaTeX
(preamble, math environments, and commands are stripped; math and citations
count as concrete anchors).
"""

import re
import statistics
from pathlib import Path

from .findings import Finding

MATH_ENVS = r"equation|align|gather|eqnarray|multline|algorithm|algorithmic|figure|table|tabular|array"
# Humans are lumpy: sentence-length CV in human prose sits around 0.5-0.6,
# instruction-tuned LLM output around 0.3-0.4. Flag below 0.40.
SENT_CV_FLOOR = 0.40
PARA_CV_FLOOR = 0.35
ANCHOR = "⎇"  # placeholder for stripped math/citations: counts as concrete


def _strip_tex_line(line: str) -> str:
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = re.sub(r"\$\$?[^$]*\$\$?", ANCHOR, line)
    line = re.sub(r"\\(?:cite[pt]?|eqref|ref|autoref|label)\*?\{[^}]*\}", ANCHOR, line)
    line = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", line)
    line = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", line)
    line = re.sub(r"\\[a-zA-Z]+\*?", "", line)
    return line


def paragraphs(path: Path) -> list[tuple[int, int, str]]:
    """Return (start_line, end_line, prose_text) per paragraph, 1-based."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    is_tex = path.suffix == ".tex"
    lines = raw
    offset = 0
    if is_tex:
        for i, l in enumerate(raw):
            if "\\begin{document}" in l:
                offset = i + 1
                lines = raw[i + 1:]
                break

    paras, cur, start, end, in_env, in_fence = [], [], None, None, 0, False
    for i, line in enumerate(lines):
        n = i + offset + 1
        if not is_tex and line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if is_tex:
            in_env += len(re.findall(r"\\begin\{(?:%s)" % MATH_ENVS, line))
            opened = in_env > 0
            in_env -= len(re.findall(r"\\end\{(?:%s)" % MATH_ENVS, line))
            if opened or in_env > 0:
                if cur:
                    cur.append(ANCHOR)  # display math inside a paragraph anchors it
                continue
            line = _strip_tex_line(line)
        else:
            line = re.sub(r"`[^`]*`", ANCHOR, line)
            line = re.sub(r"\$\$?[^$]*\$\$?", ANCHOR, line)
            line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
            if re.match(r"^\s*#{1,6}\s", line):
                continue
        if line.strip():
            if start is None:
                start = n
            end = n
            cur.append(line.strip())
        elif cur:
            paras.append((start, end, " ".join(cur)))
            cur, start, end = [], None, None
    if cur:
        paras.append((start, end, " ".join(cur)))
    return paras


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z⎇(])", text)
    return [p for p in parts if len(p.split()) >= 3]


def run(path: Path) -> tuple[list[Finding], dict]:
    paras = paragraphs(path)
    findings: list[Finding] = []
    all_sents = [s for *_, p in paras for s in sentences(p)]
    word_counts = [len(s.split()) for s in all_sents]
    metrics = {"paragraphs": len(paras), "sentences": len(all_sents),
               "words": sum(len(p.split()) for *_, p in paras)}

    if len(word_counts) >= 15:
        cv = statistics.stdev(word_counts) / statistics.mean(word_counts)
        metrics["sentence_length_cv"] = round(cv, 3)
        if cv < SENT_CV_FLOOR:
            findings.append(Finding(
                0, (0, 0), "uniformity", "uniformity.SentenceLengthCV", 0.6,
                f"Sentence lengths are uniform (CV {cv:.2f} < {SENT_CV_FLOOR}); human prose is "
                f"lumpier (~0.5+). Vary: split some long sentences, fuse some short ones.",
                payload={"cv": cv}))

    para_words = [len(p.split()) for *_, p in paras if len(p.split()) >= 10]
    if len(para_words) >= 8:
        cv = statistics.stdev(para_words) / statistics.mean(para_words)
        metrics["paragraph_length_cv"] = round(cv, 3)
        if cv < PARA_CV_FLOOR:
            findings.append(Finding(
                0, (0, 0), "uniformity", "uniformity.ParagraphLengthCV", 0.6,
                f"Paragraph lengths are uniform (CV {cv:.2f} < {PARA_CV_FLOOR}) — a document-level "
                f"AI tell. Merge or split a few paragraphs.",
                payload={"cv": cv}))

    for start, _end, p in paras:
        words = p.split()
        if len(words) < 50:
            continue
        anchors = p.count(ANCHOR) + len(re.findall(r"\d", p))
        anchors += len(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}", p))
        if anchors == 0:
            findings.append(Finding(
                start, (0, 0), "genericity", "genericity.NoConcreteAnchors", 0.5,
                f"~{len(words)}-word paragraph with no numbers, names, citations, or math — "
                f"generic content is the residual AI tell. Add a concrete fact or example.",
                match=" ".join(words[:8]) + "…"))
    return findings, metrics
