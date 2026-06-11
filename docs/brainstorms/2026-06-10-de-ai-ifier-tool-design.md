---
title: de-AI-ifier — tool design brainstorm
type: feat
status: active
date: 2026-06-10
composition_mode: ai_generated
origin: docs/references/2026-10-06-bootstrap-chat.md
---

# de-AI-ifier — tool design brainstorm

## Goal and boundary

Build an **editing assistant**, not an automatic humanizer. The tool's job is to make a human's polishing pass fast: tell them *where* the AI tells are, *why* each span reads as machine-written, and offer a fix where one can be mechanized. The human stays in the loop because (a) the strongest residual tell — content-level genericity — can only be fixed by inserting real specifics, and (b) one-click rewriters optimize detector scores at the expense of prose quality, which violates the quality-preservation goal.

Hard non-goals:

- **No detector hacking.** No translation chains, no high-temperature scramblers, no per-detector adversarial tuning. We address the *causes* (the documented habits of instruction-tuned models), so improvements generalize across detectors and survive detector updates.
- **No full rewriting.** If the tool's workflow costs more time than rewriting by hand, it has failed its prime directive. Every feature must justify itself in saved minutes.
- **Not a generic style checker.** Grammarly/Vale-style general prose linting exists; we only cover AI-specific signals.

The time budget that everything serves: **report in under a minute, polishing pass in 10–20 minutes for a ~1,500-word article, with a clear stopping criterion** so the user knows when they're done instead of editing forever.

## The three-layer model (from the reference survey)

Each layer has a different measurement instrument and a different class of fix:

| Layer | What it is | Measured with | Fixable by |
|---|---|---|---|
| Lexical | Slop words, n-grams, frame phrases ("not just X, but Y", "delve", triads) | Word/regex lists from slop-forensics, Kobak excess words, Wikipedia "Signs of AI writing" | Mostly mechanical: delete, substitute, restructure — automatable |
| Structural / rhetorical | Grammar-level habits: participial clauses at 5.3× human rate, nominalizations 1.5–2×, uniform sentence/paragraph lengths, em-dash density | pybiber (Biber's 67 features per 1,000 words) vs. a baseline; custom dependency-parse detectors | Semi-mechanical: split/recast sentences, restore verbs — assisted rewrite |
| Statistical | Token-probability signature: text is a maximum-likelihood sequence; low perplexity, low burstiness | Binoculars / Fast-DetectGPT sentence-level scores | **Only genuine human token-level edits.** LLM regeneration produces another max-likelihood sequence. The tool can localize, not fix |

Plus the layer above all three: **content genericity** — vague claims, no testable numbers, no idiosyncratic examples. No instrument fully captures it, but proxies exist (claim density, number density per paragraph, hedged generalities). Fixes are human-only; the tool's job is to point at the paragraphs that need a real fact injected.

This taxonomy directly dictates the product: the tool **auto-fixes** lexical issues, **proposes** structural rewrites, and **localizes** statistical/genericity hotspots for human attention. That triage is the whole value proposition.

## Form factor: portable agentic skill over a deterministic CLI

Two layers, cleanly split:

- **Deterministic core (Python CLI, uv-managed).** All measurement and mechanical fixing lives here: scorers, rule engine, baseline profiler, report generation. Reproducible, testable, no LLM in the loop, callable by anything.
- **Agentic skill (the front end).** A portable skill definition — a markdown instruction file plus the CLI, in the emerging cross-tool skill format (works in Claude Code, Cursor, and as plain instructions for any agent runtime; no Claude-specific APIs). The agent orchestrates the *judgment* work the CLI can't do: deciding which flags matter for this article, driving the constrained-rewrite loop, and — crucially — **collecting baseline material**. Given a specific article, the agent can assemble a topical reference set (human-written pieces on the same subject/genre, or the right slice of the user's own corpus) and feed it to `baseline build`, so percentile bands compare like with like instead of averaging over all genres.

The PostHog rule still governs the split: never send an agent to do a linter's job. Anything decidable by a rule goes in the CLI; the skill handles gathering, sequencing, and rewriting.

```
deaiify report article.md          # annotated report, severity-ranked hotspots
deaiify fix article.md             # apply safe mechanical fixes (with diff preview)
deaiify fix article.md --assist    # LLM-proposed rewrites for flagged spans, accept/reject per hunk
deaiify check article.md           # pass/fail vs. baseline percentile bands (the stopping criterion)
deaiify baseline build corpus/     # build a profile from a corpus (personal or agent-collected)
```

A polishing session:

1. **Report.** All three scorers run over the markdown. Output is a single ranked hotspot list: each entry is a span, the layer(s) that flagged it, a one-line explanation, and the fix class (auto / assisted / human). Rendered as terminal report and optionally as inline `<!-- deaiify: ... -->` comments or a sidecar `.report.md`.
2. **Auto-fix sweep.** Mechanical, meaning-preserving edits applied in one diff: slop-word deletions/substitutions, "not just X, but Y" recasts, em-dash thinning. User reviews one diff, not fifty flags.
3. **Assisted pass.** For structural flags, the tool emits a *constrained* rewrite prompt per span (e.g. "split this participial tail into a finite clause; do not add new adjectives") and shows the candidate; accept/reject per hunk. Constrained prompts keep the LLM from re-introducing its own house style.
4. **Human pass.** The report's "human-required" section lists statistical hotspots (the most max-likelihood-looking sentences) and genericity flags (paragraphs with zero concrete facts). The user edits those by hand — this is where the real humanization happens, and it's deliberately concentrated into the smallest possible set of spans.
5. **Check.** Re-run scorers; show before/after deltas. Done when scores fall inside the baseline's 5th–95th percentile band. The band, not a detector verdict, is the target — we're converging on *the user's own writing distribution*, not gaming a score.

## Stand on the shoulders of giants — build vs. reuse

Reuse (all open-source, surveyed in the reference doc):

- **Lexical lists**: slop-forensics profiles (per-model over-represented words/n-grams), Kobak's excess_words.csv, the Wikipedia-derived humanizer-skill pattern list (43 patterns), NousResearch's ANTI-SLOP.md. Compile them into one rule format; per-model profiles matter because house styles differ (calibrate against the model the user actually uses).
- **Structural features**: pybiber (the literature's own instrument — same author as pseudobibeR used in the PNAS study) for the 67 Biber features; spaCy dependency parses for the custom detectors (triads, "not just X but Y", participial tails, sentence/paragraph-length CV).
- **Statistical scorer**: Binoculars (reference implementation, ICML 2024) via llama.cpp observer/performer pair, or Fast-DetectGPT as the cheap fallback. Sentence-level scores only — we need localization, not a verdict.
- **Delivery patterns to copy**: PostHog's "never send an LLM to do a linter's job" architecture (deterministic detect → LLM fix); the no_ai_slop_writing_rules voice-profile idea (corpus-derived target metrics, not just ban lists).

Build (the thin glue, a few hundred to ~2k lines of Python, uv-managed):

1. Rule compiler: external lists → one internal rule format (regex / spaCy matcher / feature threshold).
2. The three scorer adapters + unified span-level report model.
3. Baseline profiler: run the same features over the user's pre-LLM corpus, store percentile bands.
4. Mechanical fix engine: rule → transform, with diff preview.
5. Constrained-rewrite prompt templates + accept/reject loop.
6. CLI + report renderers.

Explicitly not built: our own detector, our own paraphraser, any model training (FTPO/antislop fine-tuning is out of scope — we don't control generation).

## Personalization: the baseline corpus is the killer feature

Generic ban lists make everyone converge on the same "de-AI-ed" voice — itself a detectable signature, and not *your* voice. Instead, the user feeds a one-time `baseline build` with their pre-LLM writing. From it we derive:

- Per-1,000-word percentile bands for every Biber feature → flags become "outside *your* range", not "outside English."
- Sentence-length and paragraph-length CV targets (humans are lumpy; *this* human is lumpy in a specific way).
- A personal vocabulary; slop flags get suppressed for words the user genuinely uses.
- Optional ranked tell-list: generate LLM articles on the same topics as the baseline, train a logistic regression baseline-vs-LLM, read coefficients off as a personalized, prioritized tell list (the reference doc's "nice upgrade").

If no corpus exists, fall back to the published human baselines (the PNAS parallel corpus features are on Hugging Face).

Baselines come in two flavors, and the agentic skill makes the second one cheap:

- **Personal baseline** (static, built once): the user's pre-LLM writing — defines *voice*.
- **Topical baseline** (dynamic, per article): the agent collects human-written material matched to the article's subject and genre — defines what *human writing in this register* looks like. A technical tutorial and a personal essay have legitimately different feature distributions; comparing against the right register avoids false flags. The agent gathers candidates (web search, the user's own related writing, linked references), filters out obviously LLM-generated material (the CLI's own scorers gate admission), and hands the set to `baseline build`.

## Language scope

English first; other languages (Chinese is the likely second) are a design constraint, not a current feature. The measurement layer is inherently per-language — spaCy English parses, Biber's 67 features, English slop lists — so the architecture keeps everything language-specific behind a `Language` interface: parser adapter, feature set, rule packs, sentence segmenter. The report model, baseline profiler, fix engine plumbing, CLI, and the agentic skill are language-neutral. Adding Chinese later means supplying a new adapter bundle (e.g. a Chinese spaCy/LTP pipeline, CJK-specific tells from the multilingual pattern skills surveyed in the reference doc), not touching the core. Rule packs and baselines are tagged with a language code from day one.

## Staged build (each stage independently useful)

- **Stage 1 — lexical + heuristic report (no models, instant).** Rule compiler, slop/pattern/regex detectors, uniformity metrics, ranked report, `check` against published baselines. Pure Python + spaCy. Already saves time on day one.
- **Stage 2 — structural layer + personal baseline.** pybiber integration, baseline profiler, percentile-band flagging, mechanical fix engine with diff preview.
- **Stage 3 — statistical layer.** Binoculars sentence scores via llama.cpp (local, no API), burstiness metrics, "human-required" hotspot section. Heaviest dependency, so it's optional and last; Fast-DetectGPT as the cheap path.
- **Stage 4 — full agentic skill.** Constrained rewrites per flag class with accept/reject, plus the agentic baseline collection (topical corpus assembly per article). The skill instruction file grows alongside the CLI from Stage 1 — even the lexical-only report is worth wrapping — but this stage is where the agent-side behaviors get specified and hardened.

## Risks and honest limits

- **The statistical layer can't be machine-fixed.** Any "auto-humanize" claim there would be dishonest; the design routes it to the human pass instead. This is a feature of the boundary, not a bug.
- **Arms race**: detectors now train on humanizer output. Mitigation is the design itself — we converge on a real human's writing distribution rather than on detector evasion, which is the one strategy that doesn't decay.
- **Over-correction**: stripping every flagged feature yields flat, voiceless prose (and uniform "cleanliness" is its own tell). Percentile bands rather than zero-targets, and suppression rules from the personal corpus, are the guard.
- **Expert human readers** remain accurate even against expertly humanized text when content stays generic. The genericity flags keep this in front of the user instead of pretending the tool solved it.

## Decisions so far

- **Language**: English first; architecture keeps language-specific parts behind an adapter interface so Chinese (or others) can be added without touching the core.
- **Front end**: a portable agentic skill (not tied to one agent runtime) over a deterministic Python CLI. The agent handles judgment and gathering — including collecting per-article topical baseline material — the CLI handles everything rule-decidable.

## Open questions

1. Does a pre-LLM personal corpus exist, and how large? (Determines whether Stage 2 personalization or published baselines come first.)
2. Which LLM generates the user's drafts? (Per-model slop profiles and house-style calibration depend on it.)
3. GPU/llama.cpp available locally for Binoculars, or should Stage 3 default to Fast-DetectGPT?
4. Which agent runtimes must the skill support beyond Claude Code? (Determines how strictly the skill file avoids runtime-specific features.)
