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
    return "\n\n".join(text for *_, text in paragraphs(path))


def biber_features(docs: list[tuple[str, str]], normalize: bool = True,
                   with_tokens: bool = False):
    """docs: (doc_id, prose_text) -> polars DataFrame, one row per doc.
    normalize=True gives per-1k-TOKEN rates; False gives raw counts.
    with_tokens=True also returns {doc_id: spaCy token count} — the exposure
    rates are normalized by, needed wherever counts are pooled."""
    import polars as pl
    from pybiber import CorpusProcessor

    corp = pl.DataFrame({"doc_id": [d for d, _ in docs], "text": [t for _, t in docs]})
    tokens = CorpusProcessor().process_corpus(corp, _nlp())
    from pybiber import biber
    out = biber(tokens, normalize=normalize)
    if with_tokens:
        # pybiber normalizes per 1000 NON-PUNCTUATION tokens — exposure must match
        tc = tokens.filter(pl.col("pos") != "PUNCT").group_by("doc_id").len()
        return out, dict(zip(tc["doc_id"].to_list(), tc["len"].to_list()))
    return out


SEGMENT_WORDS = 1000
MIN_SEGMENTS = 4
SEG_P_CUTOFF = 0.05
SEG_NULL_RATE = 0.10  # by construction ~10% of human segments fall outside [p5,p95]


def _segments(path: Path) -> list[dict]:
    """Paragraphs accumulated into ~SEGMENT_WORDS-word segments with line ranges."""
    segs, cur, words, start, last_end = [], [], 0, None, None
    for line, end, text in paragraphs(path):
        if start is None:
            start = line
        last_end = end
        cur.append(text)
        words += len(text.split())
        if words >= SEGMENT_WORDS:
            segs.append({"start": start, "end": last_end, "text": "\n\n".join(cur), "words": words})
            cur, words, start = [], 0, None
    if cur and segs and words < SEGMENT_WORDS // 2:
        segs[-1]["text"] += "\n\n" + "\n\n".join(cur)
        segs[-1]["words"] += words
        segs[-1]["end"] = last_end
    elif cur:
        segs.append({"start": start, "end": last_end, "text": "\n\n".join(cur), "words": words})
    return segs


def _empirical_two_sided_p(value: float, reference: "np.ndarray") -> float:
    """Two-sided tail of value in an empirical reference sample, add-one
    smoothed; floor = 2/(n+1). Used per segment against human segment rates."""
    import numpy as np
    n = len(reference)
    lo = (1 + int((reference <= value).sum())) / (n + 1)
    hi = (1 + int((reference >= value).sum())) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def segment_findings(path: Path, profile: dict) -> list[Finding]:
    """Segments vs the PROFILE's human segment-rate distributions. Feature
    significance is the MINIMUM per-segment empirical p — segments of one
    document are self-correlated, so no independence-based aggregation
    (design: segmented analysis). The out-of-band map is description only."""
    import numpy as np

    seg_data = profile.get("seg_data") or {}
    if not seg_data.get("counts"):
        return []
    segs = _segments(path)
    if len(segs) < MIN_SEGMENTS:
        return []
    rates_df = biber_features(
        [(f"seg{i:03d}", s["text"]) for i, s in enumerate(segs)]).sort("doc_id")
    human_tokens = np.asarray(seg_data["tokens"], dtype=float)
    total_words = sum(s["words"] for s in segs)

    tested = {}
    n_human = len(seg_data["tokens"])
    floor = 2.0 / (n_human + 1)
    for col, counts in seg_data["counts"].items():
        human_rates = 1000.0 * np.asarray(counts, dtype=float) / human_tokens
        p05, p95 = float(np.quantile(human_rates, 0.05)), float(np.quantile(human_rates, 0.95))
        hmin, hmax = float(human_rates.min()), float(human_rates.max())
        span = max(hmax - hmin, 1e-9)
        rates = rates_df[col].to_numpy()
        seg_p = np.array([_empirical_two_sided_p(r, human_rates) for r in rates])
        side = [(1 if r > p95 else (-1 if r < p05 else 0)) for r in rates]
        if not any(side):
            continue
        # tiebreak among floor-tied features: distance beyond the human
        # extremes, in units of the human range
        beyond = float(max(max((r - hmax) / span for r in rates),
                           max((hmin - r) / span for r in rates), 0.0))
        tested[col] = (float(seg_p.min()), seg_p, side, rates, (p05, p95), beyond)
    if not tested:
        return []
    selected = [c_ for c_, v in tested.items() if v[0] < SEG_P_CUTOFF]

    findings = []
    for col in selected:
        p_min, seg_p, side, rates, (p05, p95), beyond = tested[col]
        at_floor = p_min <= floor + 1e-12
        gloss = FEATURE_INFO.get(col, (col.split("_", 2)[-1].replace("_", " "), "", ""))[0]
        doc_rate = float(sum(r * s["words"] for r, s in zip(rates, segs)) / total_words)
        band = profile["features"].get(col)
        in_band = bool(band and band[0] <= doc_rate <= band[2])
        seg_map = "".join("↑" if s > 0 else ("↓" if s < 0 else "·") for s in side)
        outliers = []
        for i in np.argsort(seg_p):  # worst segments = lowest per-segment p
            if side[i] == 0 or len(outliers) >= 3:
                continue
            seg = segs[i]
            arrow = "↑" if side[i] > 0 else "↓"
            outliers.append(f"L{seg['start']}–{seg['end']} {arrow} {rates[i]:.1f}/1k")
        hidden = (" — document-level rate is IN band; the local deviation is invisible "
                  "without segmentation") if in_band else ""
        n_out = sum(1 for s in side if s != 0)
        sev = min(0.6 + 0.15 * max(0.0, -float(np.log10(p_min)) - 1), 0.9)
        findings.append(Finding(
            0, (0, 0), "structural", f"biber-seg:{col}", round(sev, 2),
            f"{gloss}: most extreme segment p={p_min:.3f} (min over segments, no "
            f"independence assumed; cutoff {SEG_P_CUTOFF}); {n_out}/{len(segs)} segments outside the human "
            f"segment range [{p05:.1f}–{p95:.1f}]; doc rate {doc_rate:.1f}/1k; "
            f"worst: {'; '.join(outliers)}{hidden}.",
            payload={"p": p_min, "at_floor": at_floor, "beyond": round(beyond, 2),
                     "m_tested": len(tested),
                     "doc_rate": round(doc_rate, 2), "in_band": in_band,
                     "gloss_seg": gloss, "map": seg_map, "outliers": outliers,
                     "band": [round(p05, 2), round(p95, 2)], "n_out": n_out}))
    findings.sort(key=lambda f: (f.payload["p"], -f.payload["beyond"]))
    return findings


def simulated_bands(seg_data: dict, w_tokens: int, m: int = 2000, seed: int = 0):
    """Length-matched null bands via hierarchical block bootstrap (design):
    pick a corpus document, resample ITS segments with replacement to
    ~w_tokens, pool counts to a per-1k-token rate. Document choice carries the
    between-author component; whole-segment blocks preserve within-document
    clustering (self-correlation) that Poisson noise would understate."""
    import numpy as np
    cols = list(seg_data["counts"].keys())
    C = np.array([seg_data["counts"][c] for c in cols], dtype=float)  # [F, S]
    t = np.asarray(seg_data["tokens"], dtype=float)
    docs = np.asarray(seg_data["doc"])
    by_doc = [np.where(docs == d)[0] for d in np.unique(docs)]
    rng = np.random.default_rng(seed)
    rates = np.empty((m, len(cols)))
    for j in range(m):
        segs = by_doc[rng.integers(len(by_doc))]
        picked, tok = [], 0.0
        while tok < w_tokens:
            i = int(segs[rng.integers(len(segs))])
            picked.append(i)
            tok += t[i]
        rates[j] = 1000.0 * C[:, picked].sum(axis=1) / tok
    out = {}
    for k, col in enumerate(cols):
        r = rates[:, k]
        out[col] = {"band": [float(np.quantile(r, 0.05)), float(np.quantile(r, 0.50)),
                             float(np.quantile(r, 0.95))],
                    "sd": float(r.std())}
    return out


def run(path: Path, profile: dict) -> list[Finding]:
    text = prose(path)
    feats_df, tokmap = biber_features([(path.name, text)], with_tokens=True)
    feats = feats_df.to_dicts()[0]
    w_tokens = max(1, int(next(iter(tokmap.values()))))
    seg_data = profile.get("seg_data") or {}
    flagged = []
    if seg_data.get("counts"):
        sim = simulated_bands(seg_data, w_tokens)
        basis = f"simulated @ {w_tokens} tokens"
        items = [(col, sim[col]["band"], sim[col]["sd"]) for col in sim]
        # the two non-rate features keep empirical doc bands
        for col in ("f_43_type_token", "f_44_mean_word_length"):
            if col in profile["features"]:
                items.append((col, profile["features"][col], None))
    else:  # legacy profile without seg_data: empirical bands, granularity-switched
        use_seg = (len(text.split()) < 2500 and profile.get("features_seg"))
        band_set = profile["features_seg"] if use_seg else profile["features"]
        basis = "1k-seg (empirical)" if use_seg else "doc (empirical)"
        items = [(col, band, None) for col, band in band_set.items()]

    for col, band, sd in items:
        p05, p50, p95 = band
        v = feats.get(col)
        if v is None or p05 <= v <= p95:
            continue
        if sd is not None and sd > 1e-9:
            sigma = abs(v - p50) / sd
        else:  # empirical fallback: band-width deviation as magnitude proxy
            width = max(p95 - p05, 1e-9)
            sigma = (p05 - v if v < p05 else v - p95) / width
        key = any(k in col for k in KEY_TELLS)
        sev = min(0.55 + 0.06 * sigma + (0.1 if key else 0.0), 0.95)
        arrow = "↓" if v < p05 else "↑"
        gloss, above_hint, below_hint = FEATURE_INFO.get(
            col, (col.split("_", 2)[-1].replace("_", " "), "", ""))
        hint = below_hint if v < p05 else above_hint
        msg = (f"{gloss}: {v:.1f}/1k {arrow} band [{p05:.1f}–{p95:.1f}], "
               f"median {p50:.1f}, {sigma:.1f}σ.")
        if hint:
            msg += f" Edit: {hint}."
        flagged.append((sigma, Finding(
            0, (0, 0), "structural", f"biber:{col}", round(sev, 2), msg,
            payload={"value": v, "band": [round(b, 2) for b in band], "key_tell": key,
                     "excess": round(sigma, 2), "sigma": round(sigma, 2),
                     "gloss": gloss, "arrow": arrow, "hint": hint, "band_basis": basis})))
    flagged.sort(key=lambda t: -t[0])
    out = [f for _, f in flagged[:MAX_FINDINGS]]
    out += segment_findings(path, profile)
    return out
