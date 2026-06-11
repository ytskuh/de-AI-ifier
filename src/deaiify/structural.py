"""Structural layer: Biber features vs a baseline profile's percentile bands."""

from pathlib import Path

from .findings import Finding
from .heuristics import paragraphs

# PNAS 2025 (Reinhart et al.): instruction-tuned LLMs overuse these relative
# to humans — deviations here are stronger evidence than generic features.
KEY_TELLS = ("participle", "nominalization", "phrasal_coordination",
             "that_verb_comp", "that_adj_comp", "agentless_passive", "downtoner")
MAX_FINDINGS = 12

# Plain-language gloss + edit hint for every Biber feature pybiber emits.
# Hints are directional where a generic edit exists; features that are mostly
# register/topic-driven say so instead of inventing busywork.
FEATURE_INFO = {
    "f_01_past_tense": ("past-tense verbs ('showed', 'was')",
                        "tense mix is register-driven; only act if the text never varies tense"),
    "f_02_perfect_aspect": ("perfect aspect ('has shown', 'had been')",
                            "swap some perfects for simple past/present"),
    "f_03_present_tense": ("present-tense verbs ('shows', 'is')",
                           "register-driven; math exposition is present-heavy — judge against the band"),
    "f_04_place_adverbials": ("place adverbs ('here', 'above', 'below')",
                              "in papers often 'as shown above/below' — replace some with explicit refs (Section~X)"),
    "f_05_time_adverbials": ("time adverbs ('now', 'recently', 'then')",
                             "above band often = padded narrative framing; cut empty 'now'/'recently'"),
    "f_06_first_person_pronouns": ("first person ('we', 'our', 'us')",
                                   "if below band: write 'we' more — humans in math do; LLMs avoid it"),
    "f_07_second_person_pronouns": ("second person ('you', 'your')",
                                    "above band in formal prose = chatbot register leaking in; rewrite impersonally"),
    "f_08_third_person_pronouns": ("third person ('he', 'they', 'their')", ""),
    "f_09_pronoun_it": ("pronoun 'it' ('it follows', 'it is expensive')",
                        "if below band: plain 'It is...' sentences read more human than nominal paraphrases"),
    "f_10_demonstrative_pronoun": ("demonstrative pronouns ('this is', 'those are')",
                                   "if below band: 'This gives...' instead of repeating the full noun phrase"),
    "f_11_indefinite_pronouns": ("indefinite pronouns ('anyone', 'something')", ""),
    "f_12_proverb_do": ("pro-verb 'do' ('we do so', 'as others did')",
                        "if below band: substitute 'do so' for a repeated verb phrase occasionally"),
    "f_13_wh_question": ("direct questions ('What drives this?')",
                         "an occasional real question is a human move; LLM prose avoids or formulaizes them"),
    "f_14_nominalizations": ("verbs turned into abstract nouns ('utilization', 'convergence of...')",
                             "restore the verb: 'the estimation of X' -> 'estimating X' or 'we estimate X'"),
    "f_15_gerunds": ("-ing forms used as nouns ('sampling', 'training')",
                     "recast some as finite verbs — but domain terms ('sampling') inflate this; judge per case"),
    "f_16_other_nouns": ("overall noun density",
                         "above band = noun-stacked prose; convert some noun phrases back to clauses"),
    "f_17_agentless_passives": ("passives with no agent ('is computed')",
                                "humans use MORE of these in methods prose; if below band, some actives can relax to passive"),
    "f_18_by_passives": ("by-passives ('was proposed by X')", ""),
    "f_19_be_main_verb": ("'be' as main verb ('X is Y')",
                          "if below band, the text avoids plain 'is' — undo copula dodges ('serves as' -> 'is')"),
    "f_20_existential_there": ("existential 'there' ('there exists', 'there are')",
                               "if below band: 'There is a unique...' is normal math prose; use it"),
    "f_21_that_verb_comp": ("that-clauses after verbs ('we show that...')",
                            "if below band: restore 'that' constructions — 'We prove that...', 'This implies that...'"),
    "f_22_that_adj_comp": ("that-clauses after adjectives ('clear that...', 'likely that...')",
                           "if below band: 'It is clear that...' constructions read human in math prose"),
    "f_23_wh_clause": ("wh-clauses ('what this means', 'how X behaves')", ""),
    "f_24_infinitives": ("infinitives ('to compute', 'to show')", ""),
    "f_25_present_participle": ("participial clauses ('Running the solver, we...')",
                                "the #1 LLM tell (5.3x human) — convert to finite clauses or split the sentence"),
    "f_26_past_participle": ("past-participial clauses ('Given the data, ...')",
                             "convert some to finite clauses"),
    "f_27_past_participle_whiz": ("reduced relatives ('the model trained on X')",
                                  "expand a few: 'the model that was trained on X', or rephrase"),
    "f_28_present_participle_whiz": ("reduced relatives ('the term involving X')",
                                     "expand a few to full relative clauses"),
    "f_29_that_subj": ("'that' relative clauses on subjects ('the map that defines...')", ""),
    "f_30_that_obj": ("'that' relative clauses on objects ('the bound that we derive')", ""),
    "f_31_wh_subj": ("'which/who' relatives on subjects ('the method, which converges...')", ""),
    "f_32_wh_obj": ("'which/who' relatives on objects ('the model which we trained')", ""),
    "f_33_pied_piping": ("preposition-fronted relatives ('the rate at which...')",
                         "above band = formal register; a stranded alternative ('...which we converge at') is more colloquial"),
    "f_34_sentence_relatives": ("sentence relatives (', which means that...')",
                                "above band is an LLM elaboration habit — split into a new sentence instead"),
    "f_35_because": ("'because' clauses",
                     "if below band: plain 'because' beats 'due to the fact that' and 'as a result of'"),
    "f_36_though": ("'though/although' clauses", ""),
    "f_37_if": ("'if/unless' conditionals", ""),
    "f_38_other_adv_sub": ("other subordinators ('since', 'while', 'whereas')", ""),
    "f_39_prepositions": ("preposition density ('of', 'in', 'for')",
                          "if below band, noun phrases may be over-compressed; if above, unwind 'of'-chains"),
    "f_40_adj_attr": ("attributive adjectives ('stochastic differential equation')",
                      "dense technical noun phrases inflate this — mostly topic-driven, edit only obvious adjective stacking"),
    "f_41_adj_pred": ("predicative adjectives ('X is expensive')",
                      "if below band, more plain 'X is <adj>' statements would read more human"),
    "f_42_adverbs": ("overall adverb density",
                     "above band: cut the empty intensifying/stance adverbs first ('significantly', 'effectively')"),
    "f_43_type_token": ("vocabulary diversity (type/token ratio)",
                        "above band = unusually varied wording (elegant variation is an AI tell); repeat your terms consistently"),
    "f_44_mean_word_length": ("average word length",
                              "above band = latinate/formal vocabulary; prefer shorter everyday words where meaning allows"),
    "f_45_conjuncts": ("linking adverbs ('however', 'therefore', 'thus')",
                       "if below band, a few explicit connectives are fine — humans use them"),
    "f_46_downtoners": ("downtoners ('barely', 'nearly', 'slightly')",
                        "model families diverge from humans here (PNAS); match the band in either direction"),
    "f_47_hedges": ("hedges ('maybe', 'roughly', 'sort of')",
                    "if below band: honest hedging reads human; LLM prose either over-hedges formulaically or asserts flatly"),
    "f_48_amplifiers": ("amplifiers ('extremely', 'absolutely')",
                        "above band: delete most — they add no information"),
    "f_49_emphatics": ("emphatics ('really', 'indeed', 'in fact')",
                       "above band: cut; below band: an occasional 'in fact' is human"),
    "f_50_discourse_particles": ("discourse particles (sentence-initial 'well', 'now')", ""),
    "f_51_demonstratives": ("demonstrative determiners ('this method', 'these results')", ""),
    "f_52_modal_possibility": ("possibility modals ('can', 'may', 'might')", ""),
    "f_53_modal_necessity": ("necessity modals ('must', 'should', 'has to')",
                             "soften or vary a few ('must run until mixing' -> 'runs until mixing')"),
    "f_54_modal_predictive": ("predictive modals ('will', 'would')", ""),
    "f_55_verb_public": ("public verbs ('say', 'report', 'claim')", ""),
    "f_56_verb_private": ("private verbs ('think', 'believe', 'expect')",
                          "if below band: 'we believe/expect' statements read human; LLMs avoid committing"),
    "f_57_verb_suasive": ("suasive verbs ('propose', 'suggest', 'recommend')", ""),
    "f_58_verb_seem": ("'seem/appear'",
                       "if below band: 'this appears to...' is a human epistemic move"),
    "f_59_contractions": ("contractions ('don't', 'it's')",
                          "register-bound: fine in notes/blogs, rare in papers — judge by your venue, not the band alone"),
    "f_60_that_deletion": ("omitted 'that' ('we know ∅ it converges')",
                           "if below band: dropping 'that' occasionally is a human informality"),
    "f_61_stranded_preposition": ("stranded prepositions ('the case we care about')",
                                  "below band = formal avoidance; stranding a few reads more human"),
    "f_62_split_infinitive": ("split infinitives ('to accurately compute')",
                              "humans split infinitives; if at zero with everything else formal, that's the tell pattern"),
    "f_63_split_auxiliary": ("split auxiliaries ('are clearly shown')", ""),
    "f_64_phrasal_coordination": ("X and Y pairs of like elements",
                                  "an LLM habit when high — break some 'A and B' pairs into separate statements"),
    "f_65_clausal_coordination": ("clauses joined by and/but",
                                  "model families diverge here (PNAS); if below band, joining two short sentences with 'and' is human"),
    "f_66_neg_synthetic": ("synthetic negation ('no result', 'neither')", ""),
    "f_67_neg_analytic": ("analytic negation ('not')",
                          "if below band: LLMs under-negate; plain 'does not' statements read human"),
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
