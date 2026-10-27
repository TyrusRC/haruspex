"""Evaluation metrics (pure Python; no torch/gliner2 needed)."""

import unittest

from haruspex.eval import _mcnemar_p, _wilson, score_rows

ROWS = [{"type": "boolean", "state": "s", "question": "q", "target": True}] * 4 + \
       [{"type": "score", "state": "s", "question": "q", "levels": ["low", "mid", "high"], "target": 2}] * 2


class EvalTest(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = _wilson(80, 91)
        self.assertAlmostEqual(lo, 0.796, places=2)
        self.assertAlmostEqual(hi, 0.931, places=2)

    def test_mcnemar_only_counts_discordant_pairs(self):
        self.assertEqual(_mcnemar_p([True] * 10, [True] * 10), 1.0)
        # 10 discordant pairs, all favouring A: two-sided exact p = 2 * 0.5**10
        a = [True] * 10 + [False] * 5
        b = [False] * 10 + [False] * 5
        self.assertAlmostEqual(_mcnemar_p(a, b), 2 * 0.5 ** 10)

    def test_resolver_answers_are_reported_apart_from_the_model(self):
        def fn(row):
            if row["type"] == "boolean":
                return {"type": "boolean", "label": "yes", "confidence": 1.0, "source": "cve_lookup",
                        "accept": True, "abstain": False}
            return {"type": "score", "level": "mid", "score": 1, "confidence": 0.6, "source": "model",
                    "accept": True, "abstain": False}
        out = score_rows(fn, ROWS)
        self.assertEqual(out["by_source"]["cve_lookup"]["accuracy"], 1.0)
        self.assertEqual(out["by_source"]["model"]["accuracy"], 0.0)
        # selective coverage/risk is the calibrator's: model rows only
        self.assertEqual(out["selective"]["accepted_risk"], 1.0)
        # an off-by-one level is ordinal error 1, not just "wrong"
        self.assertEqual(out["score_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
