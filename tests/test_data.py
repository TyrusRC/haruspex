"""Schema / normalization tests — pure Python, no torch, so they run anywhere."""

import unittest

from haruspex.data import build_input, load_jsonl, normalize


class NormalizeTest(unittest.TestCase):
    def test_boolean(self):
        ex = normalize({"type": "boolean", "state": "s", "question": "q",
                        "criteria": {"true": "t", "false": "f"}, "target": True})
        self.assertEqual(ex.type, "boolean")
        self.assertEqual(len(ex.texts), 1)
        self.assertEqual(ex.target, [1.0])

    def test_boolean_soft(self):
        ex = normalize({"type": "boolean", "state": "s", "question": "q", "target": 0.3})
        self.assertAlmostEqual(ex.target[0], 0.3)

    def test_choice_onehot(self):
        ex = normalize({"type": "choice", "state": "s", "question": "q",
                        "candidates": {"a": "da", "b": "db"}, "target": "b"})
        self.assertEqual(ex.names, ["a", "b"])
        self.assertEqual(ex.target, [0.0, 1.0])

    def test_choice_soft(self):
        ex = normalize({"type": "choice", "state": "s", "question": "q",
                        "candidates": {"a": "", "b": ""}, "target": {"a": 0.8, "b": 0.2}})
        self.assertEqual(ex.target, [0.8, 0.2])

    def test_score_index(self):
        ex = normalize({"type": "score", "state": "s", "question": "q",
                        "levels": ["lo", "mid", "hi"], "target": 2})
        self.assertEqual(ex.target, [0.0, 0.0, 1.0])

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            normalize({"type": "teleport", "state": "s", "question": "q"})

    def test_choice_needs_two(self):
        with self.assertRaises(ValueError):
            normalize({"type": "choice", "state": "s", "question": "q", "candidates": {"a": ""}})

    def test_build_input_contains_parts(self):
        txt = build_input("the state", "the question", "the candidate")
        for marker in ("[STATE]", "[QUESTION]", "[CANDIDATE]", "the candidate"):
            self.assertIn(marker, txt)

    def test_seed_loads(self):
        rows = load_jsonl("data/seed.jsonl")
        self.assertGreaterEqual(len(rows), 15)
        self.assertTrue(all(r.texts for r in rows))


if __name__ == "__main__":
    unittest.main()
