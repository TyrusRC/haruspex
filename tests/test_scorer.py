"""Full per-label distributions, ensembles and conformal prediction sets (no model)."""

import math
import random
import unittest

from haruspex.calibrate import SelectiveCalibrator, _fit_set
from haruspex.model import Haruspex
from haruspex.scorer import ensemble_probs, safe_labels

CHOICE = {"type": "choice", "state": "s", "question": "q?",
          "candidates": {"a": "A", "b": "B", "c": "C"}}


class _FakeScorer:
    """Returns fixed per-member logits for every item."""

    def __init__(self, members):
        self.members, self.calls = members, 0

    def logits(self, items):
        self.calls += 1
        return [list(self.members) for _ in items]


class DistributionTest(unittest.TestCase):
    def test_every_label_gets_a_probability(self):
        v = Haruspex(None, scorer=_FakeScorer([{"a": 2.0, "b": 1.0, "c": 0.0}])).decide(CHOICE)
        self.assertEqual(v["choice"], "a")
        self.assertEqual(set(v["probabilities"]), {"a", "b", "c"})
        self.assertAlmostEqual(sum(v["probabilities"].values()), 1.0, places=5)

    def test_ensemble_disagreement_lowers_confidence(self):
        one = Haruspex(None, scorer=_FakeScorer([{"a": 5.0, "b": 0.0, "c": 0.0}])).decide(CHOICE)
        ens = Haruspex(None, scorer=_FakeScorer([{"a": 5.0, "b": 0.0, "c": 0.0},
                                                 {"a": 0.0, "b": 5.0, "c": 0.0},
                                                 {"a": 5.0, "b": 0.0, "c": 0.0}])).decide(CHOICE)
        self.assertEqual(ens["choice"], "a")
        self.assertLess(ens["confidence"], one["confidence"] - 0.2)
        self.assertAlmostEqual(ens["agreement"], 2 / 3, places=5)

    def test_one_batch_for_many_rows(self):
        sc = _FakeScorer([{"a": 1.0, "b": 0.0, "c": 0.0}])
        Haruspex(None, scorer=sc).decide_many([CHOICE] * 5)
        self.assertEqual(sc.calls, 1)

    def test_legacy_temperature_scope_is_boolean_only(self):
        # a calibrator.json written before full logits existed has no temperature_scope
        import json, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibrator.json")
            with open(path, "w") as fh:
                json.dump({"threshold": 0.0, "target_risk": 0.1, "temperature": 3.0}, fh)
            cal = SelectiveCalibrator.load(path)
        self.assertEqual(cal.temperature_for("choice"), 1.0)
        self.assertEqual(cal.temperature_for("boolean"), 3.0)


class SafeLabelTest(unittest.TestCase):
    def test_reserved_markers_are_mapped_back(self):
        safe, back = safe_labels({"low (cvss<4)": "a (b)", "[P]x": "", "low cvss<4": "z"})
        self.assertFalse(any(c in k + (v or "") for k, v in safe.items() for c in "()"))
        self.assertEqual(sorted(back.values()), ["[P]x", "low (cvss<4)", "low cvss<4"])
        self.assertEqual(len(safe), 3)   # collisions stay distinct


class PredictionSetTest(unittest.TestCase):
    def test_coverage_holds_on_exchangeable_data(self):
        # simulate a 3-way model: true label gets p ~ U(0.2, 1); check marginal coverage
        rng = random.Random(0)
        draw = lambda: [rng.uniform(0.2, 1.0) for _ in range(2000)]
        q = _fit_set(draw()[:500], alpha=0.1)
        test = draw()
        covered = sum(1 - p <= q for p in test) / len(test)
        self.assertGreaterEqual(covered, 0.88)

    def test_too_few_rows_give_no_set(self):
        self.assertIsNone(_fit_set([0.9] * 5, alpha=0.1))   # needs ceil(6*0.9)=6 > 5

    def test_set_lists_likely_labels_first(self):
        cal = SelectiveCalibrator(0.0, 0.1, set_q=0.7)
        self.assertEqual(cal.prediction_set({"a": 0.5, "b": 0.35, "c": 0.15}), ["a", "b"])

    def test_verdict_carries_the_set(self):
        cal = SelectiveCalibrator(0.99, 0.1, set_q=0.7)
        v = Haruspex(None, calibrator=cal,
                     scorer=_FakeScorer([{"a": math.log(5), "b": math.log(3.5), "c": math.log(1.5)}])).decide(CHOICE)
        self.assertTrue(v["abstain"])                 # top-1 not confident enough...
        self.assertEqual(v["prediction_set"], ["a", "b"])   # ...but the set narrows it


class EnsembleProbsTest(unittest.TestCase):
    def test_unscored_member_poisons_the_mean(self):
        self.assertIsNone(ensemble_probs([{"a": 1.0, "b": 0.0}, None]))


if __name__ == "__main__":
    unittest.main()
