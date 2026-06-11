---
title: de-AI-ifier — concrete architecture
type: design
status: active
date: 2026-06-10
composition_mode: ai_generated
origin: docs/brainstorms/2026-06-10-de-ai-ifier-tool-design.md
---

# de-AI-ifier — concrete architecture

Two layers: a deterministic Python CLI (`deaiify`, uv-managed) that does measurement and mechanical fixing, and a portable agentic skill that orchestrates judgment work (rewrites, baseline collection). Dependency facts verified against upstream repos/PyPI, June 2026.

Working principles for this build:

- **Performance first, licensing later.** Use the best available component regardless of license; anything with a restrictive/missing license is marked ⚠ in the debt list below and swapped only if/when the tool is distributed.
- **No speculative structure.** Start flat; split modules, add abstractions, or introduce interfaces only when a second concrete user of them exists. The stages below are strictly incremental — each ships something usable and nothing needed only by a later stage.

## Components (verified June 2026)

| Layer | Tool | Source / version | Use |
|---|---|---|---|
| Lexical engine | **Vale** | v3.14.2, `vale-cli/vale` (Go binary, subprocess, `--output=JSON`) | the lexical runtime |
| Lexical rules | **tbhb/vale-ai-tells** | v1.16.0 (2026-06), 50+ rules, cites Reinhart PNAS 2025 | consume via `vale sync` |
| Lexical rules | **JMill/deslop** | v0.1.0 (2026-06), ~17 rules | consume via `vale sync` |
| Lexical rules | **Nous ANTI-SLOP.md** (`NousResearch/autonovel`) | 3 tiered ban tables + 9 structural patterns, parse directly | ⚠ unlicensed |
| Lexical rules | **Wikipedia "Signs of AI writing"** | phrase lists per section | ⚠ CC BY-SA if redistributed |
| Lexical word lists | **Kobak excess_words.csv** (`berenslab/llm-excess-vocab`, MIT) | 900 words, filter `type=style` | compile → generated Vale style |
| Lexical n-grams | **slop-forensics** (`sam-paech/slop-forensics`, MIT) | `data/slop_list{,_bigrams,_trigrams}.json`; 10 per-model profiles | compile → generated Vale style |
| Parser | **spaCy 3.8** | `en_core_web_sm` (LAS ~90); `trf` to validate patterns | structural detectors |
| Structural features | **pybiber 0.3.1** (PyPI, MIT) | `PybiberPipeline`, per-1,000-word rates by default | feature extraction |
| Default baseline | **HAP-E corpus** (`browndw/human-ai-parallel-corpus-biber`, HF, MIT) | pre-extracted Biber matrices, 6 genres | derive default percentile bands offline |
| Statistical scorer | **ngpepin/binoculars** | llama.cpp full-logit Binoculars: logPPL/logXPPL per chunk, hotspot top-K, heatmaps, CLI + `--json` + HTTP API; CPU-viable, ~6–7 GB for a Q5 8B pair (sequential load) | drive via CLI/JSON. ⚠ PolyForm-NC + bus-factor 1 — reimplement on llama-cpp-python before distribution |
| Statistical 2nd signal | **Fast-DetectGPT** (`baoguangsheng/fast-detect-gpt`, MIT, active) | curvature method; GPT-Neo-2.7B cheapest config | optional cross-check, vendor the scoring function |
| Rewrite LLM | host agent of the skill | — | CLI never calls an LLM |

### ⚠ License debt (deliberate, revisit before any distribution)

1. `ngpepin/binoculars` — PolyForm-Noncommercial. Fine for personal use now; the metric is from a public paper and the design is documented, so reimplementation on llama-cpp-python (MIT) is ~300 lines when needed.
2. Rules parsed from Nous ANTI-SLOP.md — repo has no license. Keep them in a separate generated file (`Deaiify/nous.yml`) so they're trivially excisable.
3. Rules derived from Wikipedia's page — CC BY-SA; same isolation strategy (`Deaiify/wikipedia.yml`).

## Statistical layer: diversified model ensemble

Don't anchor the probability reference to a single model family. Detection strength tracks distribution match between scorer and generator, and families have distinct house styles (PNAS finding), so a single-family scorer over-flags that family's style and under-flags others.

- **Pairs, not pools**: the Binoculars metric requires observer/performer with identical tokenizers, so each scoring unit is a within-family base/instruct pair (Llama 3.1 8B, Qwen2.5 7B, Mistral 7B, …). Diversity = multiple pairs.
- **Rank aggregation, not score averaging**: B-scores aren't comparable across pairs. Each pair ranks sentences by machine-likeness; aggregate by mean rank or "top-K in ≥2 pairs". Consensus hotspots are high-confidence; single-family flags are advisory.
- **Match one pair to the drafting model**: the most informative reference is the family that generated the draft. Config = drafting-model pair + 1–2 diverse pairs.
- **Incremental**: ship with one pair; the config is already a list, adding a pair is a config entry, not code. Memory stays flat (pairs load sequentially); wall-clock scales linearly — acceptable since this runs once per polishing session, not per keystroke.
- **Pair ensemble status (2026-06-11)**: 5 pairs in `models/pairs.json`. VALIDATED: `gpt5chat-gad` and `gpt5chat-sft` (both 160k real GPT-5-Chat responses on Llama-3.1-8B; held-out GPT text scores B 0.79/0.84 + Δ>0 vs ≥0.99 + Δ<0 for Claude/human text — volume SFT on real production prose works; GAD gives deeper B dips, SFT gives calibrated Δ — keep both). NEGATIVE RESULT: `qwen35-claude46` (Jackrong 3k-sample reasoning LoRA) does not separate Claude prose at all — small reasoning-trace LoRAs learn the `<think>` format, not the prose distribution; this rules out the entire TeichAI/lordx64-class of distills as performers. The Claude axis therefore has NO working performer; the evidence-backed fix is replicating the ytz20 recipe (LMSYS prompts → Claude API → volume SFT on Qwen3.5-9B-Base). Scoring text is content-tokens only (letters), with Δ quantiles + hot-token burstiness/window-concentration to distinguish localized machine passages from global register effects.
- **Validated empirically (2026-06-10, Llama-3.1-8B Q8 pair)**: text generated by the performer itself scores B≈0.50 vs ≈1.01–1.05 for human texts and for 2026-era Claude output — i.e. the pair separates its own family perfectly but does NOT flag modern cross-family LLM text (Claude prose scored *more* surprising than human L2-learner essays). Two consequences, both anticipated by the design: (a) the pair matched to the user's actual drafting model is not an optimization, it is the difference between working and not working; (b) doc-level B comparisons across registers are confounded — simple prose scores lower B than sophisticated prose regardless of authorship — so only the within-document sentence ranking and same-register comparisons are meaningful. The instruct-excess Δ (log-likelihood ratio, per the user's design idea) separated human texts (Δ ≤ −0.11/token) from suspect LLM-assisted text (−0.085) even cross-family in this small sample; worth tracking as its own signal.

## Repository organization

Flat package; split a module only when it outgrows one file.

```
de-AI-ifier/
├── pyproject.toml          # uv; extra [stat] for the statistical layer deps
├── src/deaiify/
│   ├── cli.py              # typer: report / fix / check / baseline
│   ├── findings.py         # Finding dataclass + report merge/render (split later if needed)
│   ├── lexical.py          # Vale subprocess → Findings
│   ├── structural.py       # pybiber rates vs baseline + spaCy DependencyMatcher patterns
│   ├── heuristics.py       # uniformity (length CV, punct density) + genericity (number/claim density)
│   ├── statistical.py      # [stat] ngpepin/binoculars CLI driver, ensemble rank aggregation
│   ├── baseline.py         # corpus → percentile-band profile JSON; load/store
│   └── fixes.py            # AUTO transforms → unified diff
├── rulepacks/en/           # vale.ini + generated Deaiify style (committed, regenerable)
├── profiles/en-default.json
├── tools/build_rulepacks.py  # pinned upstream sources → Deaiify/*.yml (one file per source)
├── skills/deaiify/SKILL.md
└── tests/                  # golden-file fixtures per scorer
```

Deliberately absent until proven necessary: a `lang/` adapter layer (language-readiness = a `lang` tag on rulepacks/profiles and no hardcoded English in `findings.py`; adapters get built when Chinese work actually starts), per-model slop packs (the aggregated lists first; per-model only if flags prove too generic), an HTTP API, prompt-template library (templates live inline in SKILL.md until there are too many).

## Data flow

Every scorer emits `Finding(span, layer, rule, severity, message, fix_class, payload)`; report, fix engine, and skill all consume that one shape.

```
article.md ─┬ lexical.py      Vale JSON (ai-tells, Deslop, Deaiify)
            ├ structural.py   pybiber rates vs profile bands + dependency patterns
            ├ heuristics.py   length CV, em-dash density, per-paragraph fact density
            └ statistical.py  [opt] per-sentence rank across model pairs
                    ↓
            findings.py: dedupe overlaps, rank, attach fix class
                    ↓
   report (rich terminal / .report.md / .report.json)
   fix    (AUTO findings → one reviewable unified diff)
   check  (layer scores vs profile bands → pass/fail + deltas)
```

Severity is baseline-relative: Biber features flag only outside the profile's 5th–95th percentile band; lexical hits on words the personal profile shows the user genuinely uses are suppressed; statistical output is a ranking ("your N most machine-like sentences"), never a verdict — published thresholds don't transfer across model pairs.

## The agentic skill

`skills/deaiify/SKILL.md`: plain markdown + CLI invocations, portable across agent runtimes (no runtime-specific APIs). Encodes:

1. **Session loop**: `report` → apply `fix` diff → span-by-span constrained rewrites for ASSIST findings ("split this participial tail into a finite clause; do not add adjectives; preserve all facts") → present HUMAN findings (statistical hotspots, genericity flags) for manual editing → `check`; stop when bands pass.
2. **Topical baseline collection**: gather human-written same-register material, then `deaiify baseline build --tag <topic>`. To avoid search bias — especially *agentic* search bias (the model picking papers it knows, or phrasing queries in its own idiom) — collection must be systematic, not judgment-based: define the population by declared filters (e.g. arXiv primary categories × pre-ChatGPT date window × journal-ref/DOI quality proxy), enumerate it via a relevance-free ordering, and take a seeded uniform random sample. The agent chooses *population parameters* (declared and recorded in the manifest), never individual documents. Implemented in `tools/collect_topical_baseline.py`; for post-2022 sources, additionally gate candidates through the CLI's own scorers to reject LLM-generated text.
3. **Guardrails**: never whole-article "humanize this" prompts; facts and numbers are immutable tokens.

## Stages (each ships something usable)

1. **Report**: `cli.py`, `findings.py`, `lexical.py`, `heuristics.py`, `tools/build_rulepacks.py`, minimal SKILL.md. Day-one value: ranked hotspot report, no models.
2. **Baseline + fix**: `structural.py`, `baseline.py`, `profiles/en-default.json` (from HAP-E), `fixes.py`. Band-based `check`, mechanical `fix`.
3. **Statistical**: `statistical.py` driving ngpepin/binoculars, one pair first, ensemble via config.
4. **Full skill**: assisted rewrites, topical baseline collection.

## Open questions

Personal pre-LLM corpus — exists? size? Which LLM drafts the articles (selects the matched statistical pair and, later, per-model slop pack)? Local hardware for the GGUF pairs?
