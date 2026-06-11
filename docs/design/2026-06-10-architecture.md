---
title: de-AI-ifier — architecture
type: design
status: active
date: 2026-06-10
composition_mode: ai_generated
origin: docs/brainstorms/2026-06-10-de-ai-ifier-tool-design.md
deepened: 2026-06-11
---

# de-AI-ifier — architecture

An editing assistant that localizes AI-writing tells in articles (markdown, plain
text, LaTeX) and helps remove them while preserving meaning and the author's voice.
Two layers of software: a deterministic Python CLI (`deaiify`, uv-managed) doing all
measurement, and a portable agent skill that drives judgment edits. The human is the
rewriter; the tool's job is to say where, why, and when to stop.

Working principles (agreed across the project's sessions):

- **Performance first, licensing later.** Use the best available component; anything
  license-encumbered is marked ⚠ in the debt list and kept trivially excisable.
- **No speculative structure.** Flat modules; abstractions appear when a second
  concrete user exists. Each stage ships something usable.
- **Existing public models only.** No fine-tuning, no paid generation pipelines.
  A vendor axis with no working public performer stays absent and is reported as
  *untested* — never as "clean".
- **Detection without judgment theater.** Rankings and baseline-relative bands, not
  verdicts; calibrated thresholds don't transfer across model pairs, so we never
  print a binary "AI/human" classification.
- **Averages are not enough.** Every score ships with its distribution: quantiles,
  concentration/clustering, per-sentence and per-token-class breakdowns. A clean
  mean with a rotten tail is a rotten document.

## The three layers

| Layer | Instrument | Output | Fix path |
|---|---|---|---|
| Lexical | Vale subprocess: `ai-tells` + `Deslop` packages + generated `Deaiify` style (Kobak excess words, slop-forensics lists, Nous tables) | per-span findings with severity | human rewrite; mechanical fixes exist but are **banned** (below) |
| Structural | pybiber's 67 Biber features per 1k words vs a baseline profile's p5–p95 bands | document-level rate table with ↑/↓ direction, deviation in band-widths, direction-specific edit hint | human edits a few instances of the offending construction |
| Statistical | observer/performer GGUF pairs (Binoculars B + log-likelihood-ratio Δ) | per-pair doc scores + sentence ranking + token-class distributions | human rewrites top-ranked sentences in own words; **no model-assisted rewriting ever** (model tokens lower perplexity → more machine-like; verified empirically via GPTZero regression) |

Above all three: content genericity (paragraphs with no numbers/names/citations/math)
— only the author can fix; heuristic flags only.

## CLI surface

```
deaiify report <file> [--profile NAME] [--stat] [--min-severity X] [--json]
deaiify stat <file> [--pair NAME|all] [--classes] [--consensus] [--top N]
deaiify baseline build --name NAME <files-or-dirs>
deaiify stat --calibrate <files-or-dirs>      # per-pair human bands, see below
deaiify fix <file>                            # BANNED: warns and exits
```

Decisions made and to hold:

- **`check` is removed.** `report` is the single reading surface; a separate
  pass/fail gate duplicated it with arbitrary thresholds and ignored the
  statistical layer. Stopping criterion = report shows rates inside bands and no
  consensus statistical outliers.
- **`fix` is banned until precision work is done.** The mechanical engine produced
  three real bugs during development (multi-word collision, POS-changing swap,
  comparative-context swap); the guards caught classes, not instances. The command
  now emits a warning and refuses. Code stays for future re-enable behind that
  warning; the lexical findings still tell the human what to edit.
- The startup hints (`[setup]` lines) point fresh checkouts at README setup; they
  stay silent when everything exists.

## Statistical layer design

**Pairs** are configured in `models/pairs.json`: named (observer, performer) GGUF
file pairs with an `axis` description. Observer = pretrained base (or nearest);
performer = instruct/distill whose distribution approximates a target vendor.
Plain-text tokenizer identity is asserted at load. Current ensemble: `gpt5chat-gad`
(default — strongest validated separation), `gpt5chat-sft` (best-calibrated Δ),
`llama31-rlhf`, `qwen35-rlhf`, `qwen35-claude46` (known-weak, kept as documentation
of the negative result).

**Scores.** Per token: logprob under both models; Δ = lp_perf − lp_obs; xent =
−Σ_v P_obs(v)·log P_perf(v) (formula matched to the ahans30/Binoculars reference).
Per sentence and per document (content tokens only — tokens containing a letter):
B = logPPL_perf / xent (low = machine-typical), Δ mean and q10/50/90, hot-token
(Δ > doc q90) burstiness and max 50-token window share, and per-token-class
distributions (spaCy POS classes + curated transition list + punctuation/symbols).
Vendor fingerprint: all-classes-negative Δ profile = "not this vendor".

**Evaluation units and thresholds.** Short sentences carry too much sampling
variance for B (a 6-token sentence reaches extreme scores on noise), but no
sentence may be silently skipped. Scoring therefore operates on UNITS:
consecutive prose sentences are merged forward until a unit reaches ≥12 words;
a trailing remainder joins the previous unit. Every prose sentence belongs to
exactly one unit. (Mostly-non-letter "sentences" — leaked math/markup, <50%
letter characters — are extraction artifacts, not prose, and are excluded from
units.) Unit boundaries are computed from words of the shared cleaned text, so
they are identical across tokenizer families and consensus can aggregate by
unit. Flagging is THRESHOLD-based, not top-K: calibration stores unit-level
human bands per pair (p1/p5/p50 of B over all baseline-corpus units, p95/p99 of
Δ); a unit is flagged when B falls below the human p5 (strong: p1) or Δ rises
above p95. However many units cross the threshold is how many findings there
are — zero on a clean document. Without calibration bands the layer reports
document scores and a ranking only (explicitly labeled uncalibrated).

**Calibration.** `deaiify stat --calibrate <corpus>` scores corpus documents on
pairs and stores per-pair human bands (doc-level B and Δ p5/p50/p95, unit-level
B p1/p5/p50 and Δ p50/p95/p99) in `models/stat-bands.json`, merging per pair so
adding a model later calibrates only that pair. Profiles record the corpus
paths they were built from (`corpus_paths`), and `report --stat --profile X`
AUTO-CALIBRATES a missing pair against that profile's corpus before scoring —
one-time per pair, with a console notice (calibration is minutes, not seconds).
The bare `stat` command never auto-calibrates (it has no profile context);
uncalibrated pairs there fall back to the labeled ranking. Scores are annotated
inside/below the human band, making the statistical layer baseline-relative
like the structural layer.

**Consensus (new).** `stat --consensus` aggregates sentence rankings across all
available pairs: per sentence, mean normalized B-rank and the count of pairs with
Δ>0. Sentences machine-like on multiple independent vendor axes are the priority
rewrites; low-B with Δ≤0 everywhere is generic boilerplate (human-normal).

**Efficiency (new).** Within one document scoring run: tokenize once per tokenizer
family; evaluate each unique model once (cache per-token logprobs + fp16
log-softmax in RAM); derive every pair's B/Δ/xent from the cached arrays. This
cuts a 5-pair run from 10 model loads to one per unique model (currently 7).
Chunking at n_ctx uses a 256-token overlap; scores are taken only for tokens with
at least that much context.

## Report output organization

The report is organized for direct editing — one pass of document context, then
one top-to-bottom worklist:

1. **Document profile** (a single document-wide section): the structural rate
   table (feature, rate, ↑/↓, band, deviation, hint); a segment-heterogeneity
   TABLE — one row per uneven feature: significance, doc rate, in-band marker,
   a one-character-per-segment map (↑/·/↓ by residual) showing where in the
   document the feature swings, and the worst segment line ranges; rhythm/
   uniformity metrics; and — with --stat — per-pair document scores with band
   verdicts. These are context
   and direction-setting, not individually editable spans; there are no other
   document-level sections.
2. **Recurring patterns**: a compact rule→count summary line (no examples), so
   repeated tells are visible without duplicating the worklist.
3. **Edit worklist**: every location-bound finding from all layers (lexical
   spans, threshold-flagged statistical sentences, genericity paragraphs) in one
   table sorted by line number — the intended use is editing the document top to
   bottom alongside it.

## Segmented structural analysis

A long article can be in-band on every whole-document rate while one part runs far
outside human range — averages hide localized machine style. The segmented
analysis compares every ~1,000-word segment against the PROFILE's
segment-granularity bands (`features_seg`), not against the document's own rate:
the reference for "abnormal" is always human writing, and because the bands are
built from same-sized human segments, segment sampling noise is already inside
the band — no separate count model is needed.

- **Segmentation**: paragraphs accumulated into ~1,000-word segments (the same
  segmentation the profile's segment bands were built from). Requires ≥4
  segments; shorter documents are covered by the whole-document comparison.
- **Per-cell check**: a segment is out-of-band for a feature when its per-1k rate
  falls outside the profile's segment band [p5, p95]. By construction ~10% of
  human segments fall outside per feature, so single cells mean nothing.
- **Feature selection with multiplicity control**: per feature, the number of
  out-of-band segments under the null is Binomial(S, 0.10); the feature's
  p-value is the exact binomial tail, and features pass Benjamini–Hochberg at
  FDR 0.10. A feature flags only when MORE segments are outside human range
  than chance allows.
- **Reporting**: a table — one row per flagged feature: binomial p, document
  rate with its own in/out marker, a one-character-per-segment map (↑ above the
  human segment band / · in band / ↓ below), and the worst segments with line
  ranges and rates. The "in-band overall, locally outside" case is the hidden
  signature this exists to surface.
- **Scope note**: a document whose halves sit at opposite ends of the band but
  inside it is not flagged — in-band means within human range by definition.

## Corpora and profiles

- **Topical baseline** (`data/baseline/topical/`): pre-LLM arXiv papers collected by
  seeded uniform random sampling over declared filters (categories × pre-ChatGPT
  window × journal-ref/DOI) — the agent never picks papers, eliminating agentic
  search bias. LaTeX sources preferred (format-identical to targets). Profile:
  `profiles/topical-arxiv.json`. Straggler downloads are retried opportunistically;
  extending the sample = more draws from the same seeded stream.
- **Personal baseline**: built locally by the user from their own pre-LLM writing;
  profile `profiles/personal.json` (gitignored). The repo carries no personal file
  names, no personal corpus tooling, and no personal data — including in git
  history. Known defect fixed 2026-06-11: legacy .doc extraction leaked OOXML
  markup into six corpus files; the extraction filter now rejects markup lines and
  the profile is rebuilt.
- Profiles store p5/p50/p95 bands per Biber feature plus rhythm/lexical-density
  metrics; severity of structural findings scales with deviation measured in
  band-widths, direction-free (below band is as diagnostic as above).
- **Bands are granularity-matched.** Rate variance shrinks with text length, so
  a band computed over whole documents (8–25k words) is unfairly tight for a
  1,000-word segment or a short file. Profiles therefore carry TWO band sets:
  `features` from whole corpus documents, and `features_seg` from ~1,000-word
  segments of those documents (same segmentation as the heterogeneity
  analysis). A target is compared against the band of matched granularity:
  documents ≥2,500 prose words use document bands; shorter targets use segment
  bands; per-segment checks always use segment bands. The report labels which
  basis applied.

## Validation record (kept honest, updates appended)

- 2026-06-10: Llama-3.1-8B pair separates its own family perfectly (self-test
  B≈0.50 vs ≈1.0) and is blind to other vendors. Detection is family-specific —
  pair-to-drafting-model match is the difference between working and not working.
- 2026-06-11: volume SFT on real production prose works as a performer
  (GPT-5-Chat pairs: held-out GPT text B 0.79/0.84, Δ>0; human/Claude text ≥0.99,
  Δ<0). Small reasoning-trace LoRA distills do NOT shift prose distribution
  (Claude-4.6 distill inseparable from noise) — that class of model is ruled out.
- 2026-06-11: agent/LLM "humanizing" rewrites lower perplexity and worsen
  detector scores (GPTZero) — the basis for the no-model-rewrites rule.

## Testing policy

Only tests that lock in bugs that actually happened or invariants that broke once:
the rulepack table parser (Nous section-leak), the fix engine's three historical
bugs (kept for the future re-enable), and LaTeX prose extraction (math/preamble
stripping, markup-junk rejection). No coverage-driven test generation.

## ⚠ License debt (revisit before any formal distribution)

1. Rules parsed from Nous ANTI-SLOP.md (`rulepacks/en/Deaiify/Nous*.yml`) — upstream
   has no license; isolated in their own files, excisable by deletion.
2. ngpepin/binoculars — PolyForm-NC; **approach only** (sequential GGUF loading,
   hotspot localization); none of its code is used.
3. Model weights downloaded by users carry their own licenses; none are
   redistributed.

## Open / future

- Claude detection axis: no working public performer exists (2026-06); revisit when
  a volume prose distill appears. Until then verdicts say "Claude untested".
- Re-enable `fix` after span-safety work (grapheme-safe spans, POS-checked swaps,
  golden corpus green).
- Per-model lexical packs (slop-forensics per-model profiles) once the user's
  drafting model is known.
- Chinese support = new Language adapter bundle (parser, rule packs, segmenter);
  rule packs and profiles already carry a language tag.
