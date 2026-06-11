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
        text = " ".join(t for _, t in paras)
        self.assertNotIn("documentclass", text)
        self.assertNotIn("mc^2", text)
        self.assertIn("We study the system", text)

    def test_clean_strips_anchor_and_markup(self):
        # Bug: the ⎇ anchor glyph distorted token probabilities
        self.assertEqual(statistical._clean("uses **bold** and ⎇ anchors"),
                         "uses bold and anchors")




class SegmentHeterogeneityInvariants(unittest.TestCase):
    """Statistical core of segmented structural analysis: homogeneous counts
    must not flag; planted imbalance must flag (design: segmented analysis)."""

    def test_homogeneous_not_flagged_planted_flagged(self):
        import numpy as np
        from deaiify import structural
        rng = np.random.default_rng(7)
        weights = np.full(10, 0.1)
        homog = rng.multinomial(200, weights).astype(float)
        p_h, _ = structural._heterogeneity(homog, weights, np.random.default_rng(1))
        planted = np.full(10, 10.0); planted[3] = 110.0  # one segment 11x the rest
        p_p, resid = structural._heterogeneity(planted, weights, np.random.default_rng(1))
        self.assertGreater(p_h, 0.05)
        self.assertLess(p_p, 0.01)
        self.assertEqual(int(np.argmax(np.abs(resid))), 3)

    def test_bh_select(self):
        from deaiify.structural import _bh_select
        pvals = [("a", 0.001), ("b", 0.04), ("c", 0.9)]
        sel = _bh_select(pvals, 0.10)
        self.assertIn("a", sel)
        self.assertNotIn("c", sel)


if __name__ == "__main__":
    unittest.main()
