# de-AI-ifier

Editing assistant that localizes and fixes AI-writing tells in articles — lexical slop,
structural habits, statistical signature — while preserving meaning and the author's
voice. Design: [docs/design/2026-06-10-architecture.md](docs/design/2026-06-10-architecture.md).

## Status: Stage 3 (report + baselines + check + fix + statistical layer)

```bash
uv run deaiify report <article.md|.txt|.tex> [--profile topical-arxiv] [--stat [fast|full]] [--json]
uv run deaiify stat <article> [--pair NAME|all|fast] [--classes] [--consensus] [--top N]
uv run deaiify stat --calibrate <human-corpus>  # per-pair human bands for stat scores
uv run deaiify baseline build --name <tag> <files-or-dirs>
# `fix` is DISABLED pending precision work; `check` was removed — report is the
# single reading surface (see docs/design for both decisions)
```

The statistical layer scores text under observer/performer
GGUF pairs configured in `models/pairs.json` — currently: Llama-3.1 RLHF, Qwen3.5
RLHF, and two validated GPT-5-Chat performers (trained on 160k real ChatGPT
responses; held-out GPT text separates at B≈0.8 vs ≈1.0 for human/other text).
Per pair and per sentence: Binoculars B (low = machine-typical) and the
log-likelihood-ratio Δ (high = leans toward that performer's vendor distribution),
computed on content tokens only, with Δ quantiles, hot-token clustering, and
`--classes` score distributions by token class (POS/transitions/punctuation).
Detection is family-specific: a pair only flags text from a similar model — absence
of a vendor pair (e.g. Claude) means that axis is simply untested, not clean.
Output is a ranking to hand-rewrite — by design there is no auto-fix here: replacing
tokens with any model's preferred tokens lowers perplexity and makes text MORE
machine-like. Only human token choices fix this layer.

Profiles in `profiles/`: `topical-arxiv` (13 pre-LLM math/stat journal papers, sampled
bias-free from arXiv — see `tools/collect_topical_baseline.py`), plus any profile
you build from your own pre-LLM writing. With a profile, Biber
structural features (pybiber, 67 features) are flagged when outside the baseline's
5th–95th percentile band.

Runs Vale (packages: [vale-ai-tells](https://github.com/tbhb/vale-ai-tells),
[Deslop](https://github.com/JMill/deslop), plus the generated `Deaiify` style from
Kobak excess words / slop-forensics) and document heuristics
(sentence/paragraph uniformity, genericity), then prints one ranked finding list.
LaTeX is supported: preamble/math stripped for heuristics, en-dash rules adjusted.

## Setup

```bash
uv sync       # everything incl. llama-cpp-python (CUDA: CMAKE_ARGS="-DGGML_CUDA=on" uv sync)
              # statistical deps not wanted? uv sync --no-default-groups
mkdir -p bin && curl -sL https://github.com/vale-cli/vale/releases/download/v3.14.2/vale_3.14.2_Linux_64-bit.tar.gz | tar xz -C bin vale
uv run python tools/build_rulepacks.py    # regenerate rulepacks/en/Deaiify from upstream lists
```

The CLI prints `[setup]` hints on startup for whatever is still missing.

### Data (not in the repo — populate locally)

Baseline corpora are personal/derived and excluded from git:

- **Personal corpus**: collect your own pre-LLM writing (any .md/.txt/.tex files;
  dedup drafts, strip prompts/quotes that are not your prose) and run
  `uv run deaiify baseline build --name personal <files-or-dirs>`.
- **Topical corpus**: `uv run python tools/collect_topical_baseline.py` samples
  pre-LLM journal papers from arXiv bias-free (seeded random over declared filters;
  edit `CATEGORIES` for your field), then `sources` mode fetches LaTeX. Build with
  `uv run deaiify baseline build --name topical-arxiv data/baseline/topical/latex`.

### Models (not in the repo — ~45GB at Q8)

Detector pairs are configured in [models/pairs.json](models/pairs.json); download the
GGUFs it names into `models/` (any subset works — pairs missing files are skipped).
`tools/pget.sh <url> <out> 16` does parallel-range downloads through throttling
proxies; `hf-mirror.com` mirrors all HuggingFace URLs:

| File in pairs.json | Source (HF repo / file) |
|---|---|
| `llama3.1-8b-base.Q8_0.gguf` | `mradermacher/Meta-Llama-3.1-8B-GGUF` |
| `llama3.1-8b-instruct.Q8_0.gguf` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` |
| `qwen3.5-9b-base.Q8_0.gguf` | `mradermacher/Qwen3.5-9B-Base-GGUF` |
| `qwen3.5-9b-stock-instruct.Q8_0.gguf` | `unsloth/Qwen3.5-9B-GGUF` |
| `qwen3.5-9b-claude46-distill.Q8_0.gguf` | `Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` (weak axis — see design doc) |
| `llama3.1-8b-gpt5chat-gad.Q8_0.gguf` | `mradermacher/GAD-GPT-5-Chat-Llama-3.1-8B-Instruct-GGUF` |
| `llama3.1-8b-gpt5chat-sft.Q8_0.gguf` | `Jackrong/GPT-5-Distill-llama3.1-8B-Instruct-GGUF` |

Interpretation caveat baked into the design: a detector pair only flags text from a
*similar* model — vendor axes you haven't downloaded are untested, not clean, and no
working open Claude-prose pair exists as of 2026-06 (see the design doc's validation
notes).

Agent front end: [skills/deaiify/SKILL.md](skills/deaiify/SKILL.md).

## Principles

**AI-written text differs from human text in three measurable layers, each needing a
different instrument and a different class of fix.**

1. **Lexical** — specific words and frames appear at wildly inflated rates in LLM
   output ("delve", "not just X, but Y", "plays a crucial role"); some patterns are
   1,000× more frequent than in human text. Deterministic rules detect these exactly
   and many are mechanically fixable — never send an LLM to do a linter's job.
2. **Structural** — instruction-tuned models have grammar-level habits the author
   doesn't consciously control: present-participial clauses at ~5× the human rate,
   nominalizations at 1.5–2×, agentless passives at half, uniform sentence lengths.
   Measured with Biber's lexicogrammatical features as per-1,000-word rates and
   judged **against a baseline corpus percentile band, not a universal threshold**
   — "outside what humans in this register do", not "outside English". This also
   prevents over-correction: stripping every flagged feature flattens voice, and
   uniform cleanliness is itself a tell.
3. **Statistical** — LLM text is a maximum-likelihood token sequence. Scoring text
   under an observer/performer model pair (perplexity ratio + log-likelihood ratio)
   localizes the most machine-typical sentences. Two hard-won empirical rules:
   *detection is family-specific* — a pair only flags text from models similar to
   its performer, so verdicts are per-vendor and an untested vendor axis is never
   "clean"; and *no model can fix this layer* — any LLM rewrite (including this
   tool's own assistant) replaces tokens with model-preferred tokens, lowering
   perplexity and making text MORE machine-detectable. Only human token choices —
   concrete facts, idiosyncratic phrasing — raise surprise. The tool therefore
   ranks and localizes but never auto-rewrites here.

Above all three sits the residual tell no rewriting fixes: content genericity.
Vague claims without numbers, names, or examples read as AI to expert humans even
when every detector passes — flagged by the genericity heuristics, fixable only by
the author.

The boundary: this is an **editing assistant, not a humanizer**. It addresses the
documented causes of AI-sounding prose so improvements generalize across detectors;
it does not optimize against any specific detector's score (an arms race the
paraphrasing tools are losing — detectors train on humanizer output).

## References

- Reinhart, Brown, Markey, Laudenbach, Pantusen, Yurko & Weinberg (2025).
  *Do LLMs write like humans? Variation in grammatical and rhetorical styles.*
  PNAS 122(8):e2422455122 — the structural-layer evidence base (participials,
  nominalizations, house styles per model family).
- Hans et al. (2024). *Spotting LLMs with Binoculars: Zero-Shot Detection of
  Machine-Generated Text.* ICML 2024, [arXiv:2401.12070](https://arxiv.org/abs/2401.12070);
  reference implementation [ahans30/Binoculars](https://github.com/ahans30/Binoculars)
  (BSD-3) — the statistical layer's B score.
- Kobak et al. (2025). *Delving into LLM-assisted writing in biomedical publications
  through excess vocabulary.* Science Advances,
  [arXiv:2406.07016](https://arxiv.org/abs/2406.07016); word list at
  [berenslab/llm-excess-vocab](https://github.com/berenslab/llm-excess-vocab) (MIT).
- Biber, D. (1988). *Variation across Speech and Writing.* Cambridge UP — the 67
  lexicogrammatical features; computed via
  [pybiber](https://github.com/browndw/pybiber) (MIT).
- Paech, S. *slop-forensics* —
  [sam-paech/slop-forensics](https://github.com/sam-paech/slop-forensics) (MIT);
  over-representation analysis behind the n-gram rules (see also
  [arXiv:2510.15061](https://arxiv.org/abs/2510.15061)).
- Vale styles: [tbhb/vale-ai-tells](https://github.com/tbhb/vale-ai-tells) and
  [JMill/deslop](https://github.com/JMill/deslop) (both MIT), running on
  [Vale](https://github.com/vale-cli/vale); Wikipedia's editor-curated
  [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
  documents the pattern classes these encode.
- GPT-5-Chat detector pairs: *Black-Box On-Policy Distillation of Large Language
  Models* ([arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) — the GAD
  checkpoints and the LMSYS-prompts/GPT-5-Chat-response corpus
  (`ytz20/LMSYS-Chat-GPT-5-Chat-Response`) they and the SFT performer train on.
- Architecture precedent for local full-logit scoring with hotspot localization:
  [ngpepin/binoculars](https://github.com/ngpepin/binoculars) (PolyForm-NC —
  approach borrowed, code not used).
- This project's own validation notes (incl. the negative result on reasoning-trace
  LoRA distills as performers): [docs/design/2026-06-10-architecture.md](docs/design/2026-06-10-architecture.md).

## License

Project code is [MIT](LICENSE). The generated `Deaiify` style and all shipped rules derive
only from MIT sources (Kobak excess words, slop-forensics); `vale-ai-tells` and
`Deslop` are MIT packages fetched at runtime, not vendored.

**Optional Nous rules (not shipped).** `NousResearch/autonovel`'s ANTI-SLOP.md has
no upstream license, so its ~75 derived rules are not generated by default, are
gitignored, and are absent from this repo. They overlap heavily with the MIT
slop-forensics/Kobak lists, so omitting them barely changes coverage. To use them
locally (private use only — do not redistribute):

```bash
uv run python tools/build_rulepacks.py --include-nous
```
