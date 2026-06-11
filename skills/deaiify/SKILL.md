---
name: deaiify
description: Localize and fix AI-writing tells in an article (markdown, plain text, or LaTeX) while preserving meaning and the author's voice. Use when asked to de-AI, humanize, or polish LLM-drafted text.
---

# deaiify — polish LLM-drafted articles

Ranked detection report + baseline profiles + statistical sentence ranking.
You (the agent) drive the judgment edits; the human rewrites statistical findings.

## Workflow

1. Run the report (from the repo root). Always pass a baseline profile when one
   fits the article's register (`profiles/*.json`; e.g. `topical-arxiv` for math
   papers, `personal` for the user's voice):

   ```bash
   uv run deaiify report <article> --profile topical-arxiv --json
   ```

   Plain `report` (no `--json`) renders a human-readable table; use `--json` when you
   need spans for editing. `--min-severity 0.6` filters to warnings and errors.

   With a profile, document-level `structural` findings appear (Biber feature rates
   outside the human band, worst first; `key_tell: true` = documented LLM habits).

2. NOTE: `deaiify fix` is DISABLED (unsafe mechanical edits; see design doc) and
   `check` was removed. The report IS the workflow: edit flagged spans by hand,
   re-run report, stop when rates sit inside the bands and no consensus
   statistical outliers remain.

   New profiles: `uv run deaiify baseline build --name <tag> <files-or-dirs>`
   (collect corpora systematically — see tools/collect_topical_baseline.py — never
   by hand-picking documents you "know").

3. Triage remaining findings, highest severity first:
   - **lexical** (rules from `ai-tells.*`, `Deslop.*`, `Deaiify.*`): rewrite the flagged
     span. Substitution rules carry the replacement in the message. Edit span-by-span —
     NEVER paste whole paragraphs into an LLM with "humanize this"; that re-introduces
     model house style.
   - **structural** (with a profile): rewrite a few instances of the offending
     construction, spread across the document — e.g. gerunds above band → recast some
     "-ing" clauses as finite verbs; that-complements below band → restore a few
     "We show that…" constructions; first-person below band → use "we" where natural.
     Re-run `report` after a handful of edits; do not chase the median.
   - **uniformity** (document-level): vary sentence/paragraph lengths — split a long
     sentence, fuse two short ones. Do not rephrase content while doing this.
   - **genericity**: these need a real fact, number, citation, or example from the
     author. Collect them and ASK THE USER rather than inventing specifics.

4. Rewrite constraints (apply to every edit):
   - Facts, numbers, citations, and math are immutable tokens.
   - Prefer deletion over substitution for filler ("It is worth noting that…" → delete).
   - Match the surrounding register; do not add adjectives or new claims.
   - Suggestion-level vocabulary hits ("Kobak", "SlopWords") are advisory: rewrite only
     when several cluster in one paragraph.

5. Re-run the report. Target: falling findings/1k and no remaining warnings/errors.
   Don't chase zero suggestions — over-stripping flattens voice, which is itself a tell.

## Statistical layer (Stage 3)

```bash
uv run deaiify stat <article> --classes        # all pairs in models/pairs.json
uv run deaiify stat <article> --pair gpt5chat-sft   # one vendor axis
```

Ranks sentences per detector pair: B (low = machine-typical) and Δ (high = tilted
toward that pair's vendor). Interpretation rules: detection is family-specific —
only pairs matched to the drafting model fire (a clean score on absent vendor axes
proves nothing); sentences low-B on MULTIPLE pairs with Δ>0 are the priority
rewrites; low-B with Δ≤0 everywhere is generic boilerplate (human-normal); the
--classes table says which word class carries the tilt (adjectives/transitions =
stylistic, edit those words; uniform negative across classes = not this vendor).
HARD RULE: never
"fix" these findings by rewriting with an LLM — model-chosen replacement tokens
LOWER perplexity and make the statistical signature WORSE (verified empirically:
agent-polished text scored higher on GPTZero than the original). Present the
ranked sentences to the USER for hand-rewriting in their own words; concrete
facts, numbers, and idiosyncratic phrasing are what raise B.

## Known limits

- A clean report does NOT mean undetectable; the statistical layer
  is a ranking, not a calibrated verdict (thresholds don't transfer across pairs).
- Findings on the author's own phrasing happen (e.g. formal transitions in academic
  prose). When the user says a flagged phrase is their genuine style, leave it.
