"""Structural layer: Biber features vs a baseline profile's percentile bands."""

from pathlib import Path

from .findings import Finding
from .heuristics import paragraphs

# PNAS 2025 (Reinhart et al.): instruction-tuned LLMs overuse these relative
# to humans — deviations here are stronger evidence than generic features.
KEY_TELLS = ("participle", "nominalization", "phrasal_coordination",
             "that_verb_comp", "that_adj_comp", "agentless_passive", "downtoner")
MAX_FINDINGS = 12

# Per-feature: (plain gloss, hint when ABOVE band, hint when BELOW band).
# The displayed hint matches the actual direction; "" = no generic edit exists
# (rate is genre/topic-driven). Examples are deliberately neutral, not drawn
# from any particular document.
FEATURE_INFO = {
    "f_01_past_tense": ("past-tense verbs", "", ""),
    "f_02_perfect_aspect": ("perfect aspect ('has shown', 'had been')",
                            "swap some perfects for simple tenses", ""),
    "f_03_present_tense": ("present-tense verbs", "", ""),
    "f_04_place_adverbials": ("place adverbs ('here', 'above', 'below')",
                              "often filler ('as shown above'); replace with explicit references or cut", ""),
    "f_05_time_adverbials": ("time adverbs ('now', 'recently', 'then')",
                             "cut time-setting words that add no information", ""),
    "f_06_first_person_pronouns": ("first person ('we', 'our', 'us')",
                                   "trim only if 'we' crowds the prose; it is rarely a problem",
                                   "use first person where natural — avoiding it is a machine habit in registers where humans say 'we'"),
    "f_07_second_person_pronouns": ("second person ('you', 'your')",
                                    "direct address reads chatbot-like in formal prose; rephrase impersonally", ""),
    "f_08_third_person_pronouns": ("third person ('he', 'they', 'their')", "", ""),
    "f_09_pronoun_it": ("pronoun 'it'",
                        "replace ambiguous 'it' with its referent",
                        "plain 'It is...'/'It follows...' sentences are normal human prose; allow some"),
    "f_10_demonstrative_pronoun": ("demonstrative pronouns ('this is', 'those are')",
                                   "name the referent where 'this/that' is ambiguous",
                                   "opening an occasional sentence with 'This...' instead of repeating the noun phrase reads natural"),
    "f_11_indefinite_pronouns": ("indefinite pronouns ('anyone', 'something')", "", ""),
    "f_12_proverb_do": ("pro-verb 'do' ('do so', 'did the same')",
                        "", "substituting 'do so' for a repeated verb phrase occasionally is human"),
    "f_13_wh_question": ("direct questions",
                         "many rhetorical questions is itself a formula; cut some",
                         "an occasional genuine question is a human move"),
    "f_14_nominalizations": ("verbs turned into abstract nouns ('-tion', '-ment', '-ness')",
                             "restore the verb: 'the implementation of X' -> 'implementing X'", ""),
    "f_15_gerunds": ("-ing forms used as nouns",
                     "recast some as finite verbs — but field terminology can inflate this; judge per case", ""),
    "f_16_other_nouns": ("overall noun density",
                         "noun-stacked prose; convert some noun phrases back to clauses", ""),
    "f_17_agentless_passives": ("passives with no agent ('is computed')",
                                "name the agent in a few ('was evaluated' -> 'we evaluated')",
                                "humans use these freely in technical prose; letting some actives relax to passive is fine"),
    "f_18_by_passives": ("by-passives ('was introduced by X')", "", ""),
    "f_19_be_main_verb": ("'be' as main verb ('X is Y')",
                          "", "the text avoids plain 'is' — undo copula dodges ('serves as'/'functions as' -> 'is')"),
    "f_20_existential_there": ("existential 'there' ('there is/exists')",
                               "", "'There is/exists...' is normal prose; use it"),
    "f_21_that_verb_comp": ("that-clauses after verbs ('show that...')",
                            "", "restore explicit that-clauses ('We argue that...', 'This implies that...')"),
    "f_22_that_adj_comp": ("that-clauses after adjectives ('clear that...')",
                           "", "'It is clear that...' constructions are human-normal in argument prose"),
    "f_23_wh_clause": ("wh-clauses ('what this means', 'how X behaves')", "", ""),
    "f_24_infinitives": ("infinitives ('to compute', 'to show')", "", ""),
    "f_25_present_participle": ("participial clauses ('Running the experiment, we...')",
                                "a top LLM tell (~5x human rate): convert to finite clauses or split the sentence", ""),
    "f_26_past_participle": ("past-participial clauses ('Given these results, ...')",
                             "convert some clause-initial participles to finite clauses", ""),
    "f_27_past_participle_whiz": ("reduced relatives, past ('the value obtained')",
                                  "expand a few ('the value that we obtained') or rephrase", ""),
    "f_28_present_participle_whiz": ("reduced relatives, present ('the term involving X')",
                                     "expand a few to full relative clauses", ""),
    "f_29_that_subj": ("'that' relatives on subjects", "", ""),
    "f_30_that_obj": ("'that' relatives on objects", "", ""),
    "f_31_wh_subj": ("'which/who' relatives on subjects", "", ""),
    "f_32_wh_obj": ("'which/who' relatives on objects", "", ""),
    "f_33_pied_piping": ("preposition-fronted relatives ('the rate at which')",
                         "formal register; occasionally strand the preposition instead", ""),
    "f_34_sentence_relatives": ("sentence relatives (', which means that...')",
                                "elaboration chaining is an LLM habit; split into a new sentence", ""),
    "f_35_because": ("'because' clauses",
                     "", "plain 'because' beats 'due to the fact that' and 'owing to'"),
    "f_36_though": ("'though/although' clauses", "", ""),
    "f_37_if": ("'if/unless' conditionals", "", ""),
    "f_38_other_adv_sub": ("other subordinators ('since', 'while', 'whereas')", "", ""),
    "f_39_prepositions": ("preposition density ('of', 'in', 'for')",
                          "unwind chained 'of/in/for' phrases",
                          "may indicate over-compressed noun compounds; loosen some into prepositional phrases"),
    "f_40_adj_attr": ("attributive adjectives ('a large open dataset')",
                      "largely topic-driven (technical noun phrases); edit only gratuitous adjective stacking", ""),
    "f_41_adj_pred": ("predicative adjectives ('X is small')",
                      "", "plain 'X is <adjective>' statements read human; use some"),
    "f_42_adverbs": ("overall adverb density",
                     "cut stance adverbs that add nothing ('significantly', 'effectively')", ""),
    "f_43_type_token": ("vocabulary diversity (type/token ratio)",
                        "unusually varied wording — elegant variation is an AI tell; repeat established terms consistently",
                        "very repetitive wording; vary where it does not harm precision"),
    "f_44_mean_word_length": ("average word length",
                              "latinate vocabulary; prefer shorter everyday words where meaning allows", ""),
    "f_45_conjuncts": ("linking adverbs ('however', 'therefore', 'thus')",
                       "thin them out — connect by content instead",
                       "a few explicit connectives are human-normal"),
    "f_46_downtoners": ("downtoners ('barely', 'nearly', 'slightly')",
                        "trim a few", "occasional mild downtoning reads human"),
    "f_47_hedges": ("hedges ('maybe', 'roughly', 'about')",
                    "formulaic over-hedging; commit where you are sure",
                    "honest hedging reads human; LLMs either over-hedge formulaically or assert flatly"),
    "f_48_amplifiers": ("amplifiers ('extremely', 'absolutely')",
                        "delete most — they add no information", ""),
    "f_49_emphatics": ("emphatics ('really', 'indeed', 'in fact')",
                       "cut clusters of them", "an occasional 'in fact' is human"),
    "f_50_discourse_particles": ("discourse particles (sentence-initial 'well', 'now')", "", ""),
    "f_51_demonstratives": ("demonstrative determiners ('this method')", "", ""),
    "f_52_modal_possibility": ("possibility modals ('can', 'may', 'might')", "", ""),
    "f_53_modal_necessity": ("necessity modals ('must', 'should', 'has to')",
                             "delete or soften a few where the claim stands without them", ""),
    "f_54_modal_predictive": ("predictive modals ('will', 'would')", "", ""),
    "f_55_verb_public": ("public verbs ('say', 'report', 'claim')", "", ""),
    "f_56_verb_private": ("private verbs ('think', 'believe', 'expect')",
                          "", "committed statements ('we believe', 'we expect') read human"),
    "f_57_verb_suasive": ("suasive verbs ('propose', 'suggest', 'recommend')", "", ""),
    "f_58_verb_seem": ("'seem/appear'",
                       "", "'appears to / seems to' is a human epistemic move; allow some"),
    "f_59_contractions": ("contractions ('don't', 'it's')",
                          "expand them if the venue is formal",
                          "in informal genres contractions read human; in formal papers ignore this signal"),
    "f_60_that_deletion": ("omitted 'that' ('we know it holds')",
                           "", "dropping 'that' occasionally is a human informality"),
    "f_61_stranded_preposition": ("stranded prepositions ('the case we care about')",
                                  "", "formal avoidance; stranding a few reads more human"),
    "f_62_split_infinitive": ("split infinitives ('to carefully check')",
                              "", "humans split infinitives; total avoidance plus formal register is a tell pattern"),
    "f_63_split_auxiliary": ("split auxiliaries ('are clearly shown')", "", ""),
    "f_64_phrasal_coordination": ("paired elements ('X and Y')",
                                  "an LLM habit when high — break some pairs into separate statements", ""),
    "f_65_clausal_coordination": ("clauses joined by 'and/but'",
                                  "subordinate some coordinated clauses",
                                  "joining two short sentences with 'and' occasionally is human"),
    "f_66_neg_synthetic": ("synthetic negation ('no result', 'neither')", "", ""),
    "f_67_neg_analytic": ("analytic negation ('not')",
                          "", "LLMs under-negate; plain 'does not' statements read human"),
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


def biber_features(docs: list[tuple[str, str]], normalize: bool = True):
    """docs: (doc_id, prose_text) -> polars DataFrame, one row per doc.
    normalize=True gives per-1k rates; False gives raw counts."""
    import polars as pl
    from pybiber import CorpusProcessor

    corp = pl.DataFrame({"doc_id": [d for d, _ in docs], "text": [t for _, t in docs]})
    tokens = CorpusProcessor().process_corpus(corp, _nlp())
    from pybiber import biber
    return biber(tokens, normalize=normalize)


SEGMENT_WORDS = 1000
MIN_SEGMENTS = 4
HET_BOOT = 999
HET_FDR_Q = 0.10
MIN_FEATURE_COUNT = 10


def _segments(path: Path) -> list[dict]:
    """Paragraphs accumulated into ~SEGMENT_WORDS-word segments with line ranges."""
    segs, cur, words, start = [], [], 0, None
    for line, text in paragraphs(path):
        if start is None:
            start = line
        cur.append(text)
        words += len(text.split())
        if words >= SEGMENT_WORDS:
            segs.append({"start": start, "end": line, "text": "\n\n".join(cur), "words": words})
            cur, words, start = [], 0, None
    if cur and segs and words < SEGMENT_WORDS // 2:
        segs[-1]["text"] += "\n\n" + "\n\n".join(cur)
        segs[-1]["words"] += words
    elif cur:
        segs.append({"start": start, "end": start, "text": "\n\n".join(cur), "words": words})
    return segs


def _heterogeneity(counts: "np.ndarray", weights: "np.ndarray", rng) -> tuple[float, "np.ndarray"]:
    """Max-|Pearson residual| test of multinomial homogeneity, bootstrap p-value."""
    import numpy as np
    n = int(counts.sum())
    expected = n * weights
    resid = (counts - expected) / np.sqrt(expected)
    stat = float(np.abs(resid).max())
    sims = rng.multinomial(n, weights, size=HET_BOOT)
    sim_stat = np.abs((sims - expected) / np.sqrt(expected)).max(axis=1)
    p = float((1 + (sim_stat >= stat).sum()) / (HET_BOOT + 1))
    return p, resid


def _bh_select(pvals: list[tuple[str, float]], q: float) -> set[str]:
    """Benjamini-Hochberg: names whose p-value passes FDR level q."""
    m = len(pvals)
    ordered = sorted(pvals, key=lambda t: t[1])
    keep, thresh_rank = set(), 0
    for rank, (_, p) in enumerate(ordered, 1):
        if p <= q * rank / m:
            thresh_rank = rank
    return {name for name, _ in ordered[:thresh_rank]}


def segment_findings(path: Path, profile: dict) -> list[Finding]:
    """Within-document heterogeneity per feature (design: segmented analysis)."""
    import numpy as np

    segs = _segments(path)
    if len(segs) < MIN_SEGMENTS:
        return []
    counts_df = biber_features(
        [(f"seg{i:03d}", s["text"]) for i, s in enumerate(segs)], normalize=False)
    counts_df = counts_df.sort("doc_id")
    weights = np.array([s["words"] for s in segs], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(0)

    tested = {}
    for col in counts_df.columns:
        if col == "doc_id" or col in ("f_43_type_token", "f_44_mean_word_length"):
            continue  # not counts
        counts = counts_df[col].to_numpy().astype(float)
        if counts.sum() < MIN_FEATURE_COUNT:
            continue
        p, resid = _heterogeneity(counts, weights, rng)
        tested[col] = (p, resid, counts)
    if not tested:
        return []
    selected = _bh_select([(c, v[0]) for c, v in tested.items()], HET_FDR_Q)

    findings = []
    for col in selected:
        p, resid, counts = tested[col]
        gloss = FEATURE_INFO.get(col, (col.split("_", 2)[-1].replace("_", " "), "", ""))[0]
        doc_rate = 1000 * counts.sum() / sum(s["words"] for s in segs)
        band = profile["features"].get(col)
        in_band = band and band[0] <= doc_rate <= band[2]
        outliers = []
        for i in np.argsort(-np.abs(resid)):
            if abs(resid[i]) < 2 or len(outliers) >= 3:
                break
            seg = segs[i]
            arrow = "↑" if resid[i] > 0 else "↓"
            rate = 1000 * counts[i] / seg["words"]
            outliers.append(f"L{seg['start']}–{seg['end']} {arrow}{abs(resid[i]):.1f}σ "
                            f"({rate:.1f}/1k)")
        hidden = " — document-level rate is IN band; the imbalance is invisible without segmentation" if in_band else ""
        sev = min(0.6 + 0.05 * float(np.abs(resid).max()), 0.9)
        findings.append(Finding(
            0, (0, 0), "structural", f"biber-seg:{col}", round(sev, 2),
            f"{gloss}: uneven across the document (q<{HET_FDR_Q}, p={p:.3f}); "
            f"doc rate {doc_rate:.1f}/1k; outlier segments: {'; '.join(outliers)}{hidden}.",
            payload={"p": p, "doc_rate": round(doc_rate, 2), "in_band": bool(in_band),
                     "residuals": [round(float(r), 2) for r in resid]}))
    findings.sort(key=lambda f: f.payload["p"])
    return findings


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
        arrow = "↓" if v < p05 else "↑"
        gloss, above_hint, below_hint = FEATURE_INFO.get(
            col, (col.split("_", 2)[-1].replace("_", " "), "", ""))
        hint = below_hint if v < p05 else above_hint
        msg = f"{gloss}: {v:.1f}/1k {arrow} band [{p05:.1f}–{p95:.1f}], median {p50:.1f}."
        if hint:
            msg += f" Edit: {hint}."
        flagged.append((excess, Finding(
            0, (0, 0), "structural", f"biber:{col}", round(sev, 2), msg,
            payload={"value": v, "band": band, "key_tell": key, "excess": round(excess, 2),
                     "gloss": gloss, "arrow": arrow, "hint": hint})))
    flagged.sort(key=lambda t: -t[0])
    out = [f for _, f in flagged[:MAX_FINDINGS]]
    out += segment_findings(path, profile)
    return out
