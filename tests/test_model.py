"""decide() contract with a stub encoder (pure Python; no torch/gliner2 needed)."""

import unittest

from haruspex.calibrate import SelectiveCalibrator
from haruspex.model import Haruspex


class _Stub:
    """Stands in for the gliner2 model: returns a fixed classify_text payload."""

    def __init__(self, out):
        self.out = out

    def classify_text(self, text, tasks, threshold=0.0, include_confidence=True):
        return {"decision": self.out}


BOOL = {"type": "boolean", "state": "s", "question": "q?",
        "criteria": {"true": "t", "false": "f"}}


class DecideTest(unittest.TestCase):
    def test_unscored_label_is_not_full_confidence(self):
        # a bare label with no confidence must not sail through the gate as 1.0
        cal = SelectiveCalibrator(0.5, 0.1)
        v = Haruspex(_Stub("yes"), calibrator=cal).decide(BOOL)
        self.assertEqual(v["confidence"], 0.0)
        self.assertTrue(v["abstain"])
        self.assertEqual(v["route"], "escalate")

    def test_gate_is_per_type(self):
        cal = SelectiveCalibrator(None, 0.1, by_type={"choice": {"threshold": 0.5}})
        hx = Haruspex(_Stub({"label": "yes", "confidence": 0.9}), calibrator=cal)
        self.assertTrue(hx.decide(BOOL)["abstain"])   # boolean never calibrated
        row = {"type": "choice", "state": "s", "question": "q?", "candidates": {"a": "A", "b": "B"}}
        hx = Haruspex(_Stub({"label": "a", "confidence": 0.9}), calibrator=cal)
        self.assertTrue(hx.decide(row)["accept"])

    def test_temperature_scales_boolean_probability(self):
        cal = SelectiveCalibrator(None, 0.1, temperature=2.0, by_type={"boolean": {"threshold": 0.0}})
        v = Haruspex(_Stub({"label": "no", "confidence": 0.9}), calibrator=cal).decide(BOOL)
        # softmax(log p / T) over two labels == sigmoid(logit(p) / T)
        self.assertAlmostEqual(v["probability"], 0.1 ** 0.5 / (0.1 ** 0.5 + 0.9 ** 0.5), places=5)
        self.assertLess(v["confidence"], 0.9)
        self.assertEqual(v["label"], "no")

    def test_model_answers_are_tagged(self):
        v = Haruspex(_Stub({"label": "yes", "confidence": 0.8})).decide(BOOL)
        self.assertEqual(v["source"], "model")


class TrainGuardTest(unittest.TestCase):
    def test_full_mode_refuses_before_training(self):
        # full-mode output cannot be reloaded (gliner2 2.1 save bug): fail before the
        # hours of training, not after
        import time
        from haruspex.train import train
        t0 = time.perf_counter()
        with self.assertRaises(NotImplementedError):
            train("unused-base", [], "unused-out", mode="full")
        self.assertLess(time.perf_counter() - t0, 1.0)

    def test_seed_and_lr_reach_the_trainer_config(self):
        # GLiNER2Trainer re-seeds from its own config (default 42), so a --seed that
        # is not forwarded there is silently ignored
        from unittest import mock
        try:
            import gliner2.training.trainer as gt
        except ImportError:
            self.skipTest("gliner2 not installed")
        from haruspex.train import train
        seen = {}

        def fake_trainer(model, config):
            seen["cfg"] = config
            raise StopIteration

        with mock.patch("gliner2.GLiNER2.from_pretrained"), \
                mock.patch.object(gt, "GLiNER2Trainer", fake_trainer), \
                self.assertRaises(StopIteration):
            train("unused-base", [BOOL | {"target": True}], "unused-out", seed=7, lr=3e-4)
        self.assertEqual(seen["cfg"].seed, 7)
        self.assertEqual(seen["cfg"].task_lr, 3e-4)


if __name__ == "__main__":
    unittest.main()
