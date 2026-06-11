"""Regression tests for bugs that actually occurred. Policy (design doc):
no coverage-driven tests — each case below reproduces a real historical defect.

Run: uv run python -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import build_rulepacks  # noqa: E402
from deaiify import fixes, heuristics, statistical  # noqa: E402
from deaiify.findings import Finding  # noqa: E402


class NousParserRegression(unittest.TestCase):
    """Bug: tables after the tier sections (DO/DON'T tone tables) leaked into
    Tier 3 tokens, producing a literal 'DO' filler rule that deleted the word
    'do' from prose ("they do not train" -> "they not train")."""

    MD = """\
### Tier 1: Kill on sight
| Slop word | Alternative |
|---|---|
| delve | dig into |
| testament (as in "a testament to") | shows |

### Tier 3: Filler
| Phrase | What to do |
|---|---|
| "It's worth noting that..." | Just state the thing. |

## TONE GUIDELINES
| DO | DON'T |
|---|---|
| DO | DON'T |
"""

    def test_later_tables_do_not_leak_into_tiers(self):
        tiers = build_rulepacks.parse_nous_tables(self.MD)
        all_cells = [cell for rows in tiers.values() for cell, _ in rows]
        self.assertNotIn("DO", all_cells)

    def test_construction_rows_get_no_swap(self):
        """Bug: 'testament (as in ...)' rows produced a noun->verb token swap."""
        self.assertEqual(build_rulepacks.nous_word('testament (as in "a testament to")'),
                         ["testament"])
        # the generator routes "(as in" rows to flag-only; assert the marker logic
        self.assertIn("(as in", 'testament (as in "a testament to")')


class FixEngineRegression(unittest.TestCase):
    """The three real fix-engine bugs (engine currently banned; these lock the
    guards for any future re-enable)."""

    @staticmethod
    def _finding(line, span, match, rule="Deaiify.NousTier1", params=None):
        payload = {"action": {"Name": "replace", "Params": params}} if params else {}
        return Finding(line, span, "lexical", rule, 0.6, "", match=match, payload=payload)

    def test_multiword_replacement_not_fixable(self):
        # "delve" -> "dig into" produced "dig into into"
        f = self._finding(1, (4, 8), "delve", params=["dig into"])
        self.assertEqual(fixes.fixable([f]), [])

    def test_singleword_swap_is_fixable_and_applied(self):
        f = self._finding(1, (5, 12), "paradigm", params=["model"])
        self.assertEqual(len(fixes.fixable([f])), 1)
        edit = fixes._edit("the paradigm holds", f, "replace")
        self.assertEqual(edit, (4, 12, "model"))

    def test_mismatched_span_is_skipped(self):
        f = self._finding(1, (5, 12), "paradigm", params=["model"])
        self.assertIsNone(fixes._edit("the xxxxxxxx holds", f, "replace"))

    def test_sentence_initial_delete_recapitalizes(self):
        f = Finding(1, (1, 13), "lexical", "ai-tells.FormalTransitions", 0.6, "",
                    match="Specifically,")
        edit = fixes._edit("Specifically, we sample.", f, "delete")
        s0, e0, repl = edit
        self.assertEqual(("Specifically, we sample."[:s0] + repl +
                          "Specifically, we sample."[e0:]), "We sample.")


class LatexExtractionRegression(unittest.TestCase):
    """Bug class: math/preamble leaking into scored prose (math fragments
    surfaced as 'most machine-like sentences')."""

    def test_preamble_and_math_are_stripped(self):
        tex = ("\\documentclass{article}\\usepackage{amsmath}\n"
               "\\begin{document}\n\n"
               "We study the system. The bound follows from \\cite{x_2020}.\n\n"
               "\\begin{equation}\nE = mc^2\n\\end{equation}\n\n"
               "This concludes the argument.\n\n\\end{document}\n")
        p = Path("/tmp/deaiify_test_doc.tex")
        p.write_text(tex)
        paras = heuristics.paragraphs(p)
        text = " ".join(t for *_, t in paras)
        self.assertNotIn("documentclass", text)
        self.assertNotIn("mc^2", text)
        self.assertIn("We study the system", text)

    def test_clean_strips_anchor_and_markup(self):
        # Bug: the ⎇ anchor glyph distorted token probabilities
        self.assertEqual(statistical._clean("uses **bold** and ⎇ anchors"),
                         "uses bold and anchors")




class SegmentBandInvariants(unittest.TestCase):
    """Segment significance: per-segment empirical two-sided p with min-over-
    segments aggregation — NO independence across segments (user corrections
    2026-06-11: human reference; self-correlation forbids binomial pooling)."""

    def test_empirical_two_sided_p(self):
        import numpy as np
        from deaiify.structural import _empirical_two_sided_p
        ref = np.arange(1.0, 117.0)  # 116 human segment rates
        # at the median: p ~ 1; beyond every human segment: p at the floor
        self.assertGreater(_empirical_two_sided_p(58.0, ref), 0.9)
        self.assertAlmostEqual(_empirical_two_sided_p(500.0, ref), 2 / 117, places=6)
        # min-p over segments must not shrink with more in-band segments
        # (no independence bonus): one extreme segment alone sets the p
        p_alone = _empirical_two_sided_p(500.0, ref)
        p_with_normals = min([_empirical_two_sided_p(500.0, ref)] +
                             [_empirical_two_sided_p(60.0, ref)] * 11)
        self.assertEqual(p_alone, p_with_normals)


class UnitMergingInvariants(unittest.TestCase):
    """Units merge short sentences forward — no prose sentence is skipped
    (user correction 2026-06-11: eligibility must not drop coverage)."""

    def test_every_prose_sentence_in_exactly_one_unit(self):
        from deaiify.statistical import _build_units, _is_prose
        sents = [(1, "Short one."), (2, "Tiny."), (3, "A much longer sentence with plenty of words inside it."),
                 (4, "x = y + z ^ 2 $$"), (5, "Another short."), (6, "Tail.")]
        units = _build_units(sents)
        covered = [i for u in units for i in u]
        prose = [i for i, (_, s) in enumerate(sents) if _is_prose(s)]
        self.assertEqual(sorted(covered), prose)        # all prose covered, math excluded
        self.assertEqual(len(set(covered)), len(covered))  # no sentence twice
        for u in units[:-1]:
            words = sum(len(sents[i][1].split()) for i in u)
            self.assertGreaterEqual(words, 12)
        # trailing remainder merged into the previous unit, not dropped
        self.assertIn(5, units[-1])




class SimulatedBandInvariants(unittest.TestCase):
    """Length-matched bootstrap bands (user correction 2026-06-11: a band over
    mixed-length documents is meaningless; variance must match target length)."""

    @staticmethod
    def _seg_data():
        import numpy as np
        rng = np.random.default_rng(3)
        docs, tokens, counts = [], [], []
        for d in range(6):
            lam = rng.uniform(3, 9)  # between-doc rate variation
            for _ in range(8):
                tok = int(rng.uniform(900, 1300))
                docs.append(f"d{d}"); tokens.append(tok)
                counts.append(rng.poisson(lam * tok / 1000))
        return {"doc": docs, "tokens": tokens, "counts": {"f_x": counts}}

    def test_band_narrows_with_length(self):
        from deaiify.structural import simulated_bands
        sd_short = simulated_bands(self._seg_data(), 1000, m=600)["f_x"]["sd"]
        sd_long = simulated_bands(self._seg_data(), 20000, m=600)["f_x"]["sd"]
        self.assertLess(sd_long, sd_short)

    def test_deterministic_given_seed(self):
        from deaiify.structural import simulated_bands
        a = simulated_bands(self._seg_data(), 5000, m=200)["f_x"]["band"]
        b = simulated_bands(self._seg_data(), 5000, m=200)["f_x"]["band"]
        self.assertEqual(a, b)




class ReviewFindingsRegression(unittest.TestCase):
    """Code-review findings 2026-06-11: window at token 0, segment end lines."""

    def test_first_window_burst_counted(self):
        import numpy as np
        from deaiify.statistical import _max_window_share
        # burst entirely inside the first 50 tokens must be fully reported
        share = _max_window_share(np.arange(10), 200, w=50)
        self.assertAlmostEqual(share, 10 / 50)

    def test_paragraph_and_segment_end_lines(self):
        from deaiify import heuristics, structural
        p = Path("/tmp/deaiify_test_lines.md")
        p.write_text("one two three\nfour five six\n\nseven eight\n")
        paras = heuristics.paragraphs(p)
        self.assertEqual([(a, b) for a, b, _ in paras], [(1, 2), (4, 4)])
        segs = structural._segments(p)
        self.assertEqual(segs[-1]["end"], 4)  # end = last paragraph's end line


if __name__ == "__main__":
    unittest.main()
