"""Structural layer: Biber features vs a baseline profile's percentile bands."""

from pathlib import Path

from .findings import Finding
from .heuristics import paragraphs

# PNAS 2025 (Reinhart et al.): instruction-tuned LLMs overuse these relative
# to humans — deviations here are stronger evidence than generic features.
KEY_TELLS = ("participle", "nominalization", "phrasal_coordination",
             "that_verb_comp", "that_adj_comp", "agentless_passive", "downtoner")
MAX_FINDINGS = 12

# Plain-language gloss + edit hint per Biber feature (the ones that commonly
# flag; anything missing falls back to the raw feature name).
FEATURE_INFO = {
    "f_02_perfect_aspect": ("perfect aspect ('has shown', 'had been')",
                            "swap some perfects for simple past/present"),
    "f_04_place_adverbials": ("place adverbs ('here', 'above', 'below')",
                              "in papers often 'as shown above/below' — replace some with explicit refs (Section~X)"),
    "f_06_first_person_pronouns": ("first person ('we', 'our', 'us')",
                                   "if below band: write 'we' more — humans in math do; LLMs avoid it"),
    "f_14_nominalizations": ("verbs turned into abstract nouns ('utilization', 'convergence of...')",
                             "restore the verb: 'the estimation of X' -> 'estimating X' or 'we estimate X'"),
    "f_15_gerunds": ("-ing forms used as nouns ('sampling', 'training')",
                     "recast some as finite verbs — but domain terms ('sampling') inflate this; judge per case"),
    "f_17_agentless_passives": ("passives with no agent ('is computed')",
                                "humans use MORE of these in methods prose; if below band, some actives can relax to passive"),
    "f_19_be_main_verb": ("'be' as main verb ('X is Y')",
                          "if below band, the text avoids plain 'is' — undo copula dodges ('serves as' -> 'is')"),
    "f_21_that_verb_comp": ("that-clauses after verbs ('we show that...')",
                            "if below band: restore 'that' constructions — 'We prove that...', 'This implies that...'"),
    "f_23_wh_clause": ("wh-clauses ('what this means', 'how X behaves')", ""),
    "f_25_present_participle": ("participial clauses ('Running the solver, we...')",
                                "the #1 LLM tell (5.3x human) — convert to finite clauses or split the sentence"),
    "f_26_past_participle": ("past-participial clauses ('Given the data, ...')",
                             "convert some to finite clauses"),
    "f_27_past_participle_whiz": ("reduced relatives ('the model trained on X')",
                                  "expand a few: 'the model that was trained on X', or rephrase"),
    "f_28_present_participle_whiz": ("reduced relatives ('the term involving X')",
                                     "expand a few to full relative clauses"),
    "f_29_that_subj": ("'that' relative clauses on subjects ('the map that defines...')", ""),
    "f_39_prepositions": ("preposition density ('of', 'in', 'for')",
                          "if below band, noun phrases may be over-compressed; if above, unwind 'of'-chains"),
    "f_40_adj_attr": ("attributive adjectives ('stochastic differential equation')",
                      "dense technical noun phrases inflate this — mostly topic-driven, edit only obvious adjective stacking"),
    "f_41_adj_pred": ("predicative adjectives ('X is expensive')",
                      "if below band, more plain 'X is <adj>' statements would read more human"),
    "f_43_type_token": ("vocabulary diversity (type/token ratio)",
                        "above band = unusually varied wording (elegant variation is an AI tell); repeat your terms consistently"),
    "f_44_mean_word_length": ("average word length",
                              "above band = latinate/formal vocabulary; prefer shorter everyday words where meaning allows"),
    "f_45_conjuncts": ("linking adverbs ('however', 'therefore', 'thus')",
                       "if below band, a few explicit connectives are fine — humans use them"),
    "f_53_modal_necessity": ("necessity modals ('must', 'should', 'has to')",
                             "soften or vary a few ('must run until mixing' -> 'runs until mixing')"),
    "f_61_stranded_preposition": ("stranded prepositions ('the case we care about')",
                                  "below band = formal avoidance; stranding a few reads more human"),
    "f_64_phrasal_coordination": ("X and Y pairs of like elements",
                                  "an LLM habit when high — break some 'A and B' pairs into separate statements"),
}

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


def prose(path: Path) -> str:
    return "\n\n".join(text for _, text in paragraphs(path))


def biber_features(docs: list[tuple[str, str]]):
    """docs: (doc_id, prose_text) -> polars DataFrame, one row per doc, per-1k rates."""
    import polars as pl
    from pybiber import CorpusProcessor

    corp = pl.DataFrame({"doc_id": [d for d, _ in docs], "text": [t for _, t in docs]})
    tokens = CorpusProcessor().process_corpus(corp, _nlp())
    from pybiber import biber
    return biber(tokens)


def run(path: Path, profile: dict) -> list[Finding]:
    feats = biber_features([(path.name, prose(path))]).to_dicts()[0]
    flagged = []
    for col, band in profile["features"].items():
        p05, p50, p95 = band
        v = feats.get(col)
        if v is None or p05 <= v <= p95:
            continue
        width = max(p95 - p05, 1e-9)
        # direction-free outlier magnitude: below-band is as diagnostic as above
        excess = (p05 - v if v < p05 else v - p95) / width
        key = any(k in col for k in KEY_TELLS)
        sev = min(0.6 + 0.2 * excess + (0.1 if key else 0.0), 0.95)
        side = "below" if v < p05 else "above"
        gloss, hint = FEATURE_INFO.get(col, (col.split("_", 2)[-1].replace("_", " "), ""))
        msg = (f"{gloss}: {v:.1f}/1k, {side} the human band [{p05:.1f}–{p95:.1f}] "
               f"(median {p50:.1f}).")
        if hint:
            msg += f" Edit: {hint}."
        flagged.append((excess, Finding(
            0, (0, 0), "structural", f"biber:{col}", round(sev, 2), msg,
            payload={"value": v, "band": band, "key_tell": key, "excess": round(excess, 2)})))
    flagged.sort(key=lambda t: -t[0])
    return [f for _, f in flagged[:MAX_FINDINGS]]
