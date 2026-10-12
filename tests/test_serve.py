"""Serving-layer mapping tests (pure Python; no torch/starlette needed)."""

import unittest

from haruspex.serve import _question_to_row


class QuestionMappingTest(unittest.TestCase):
    def test_boolean(self):
        r = _question_to_row("s", {"type": "boolean", "instructions": "q?",
                                   "criteria": {"true": "t", "false": "f"}})
        self.assertEqual(r["type"], "boolean")
        self.assertEqual(r["criteria"], {"true": "t", "false": "f"})

    def test_noul_is_boolean(self):
        # TypeSafe's systemone API calls boolean "noul" — must map to boolean.
        r = _question_to_row("s", {"type": "noul", "instructions": "q?"})
        self.assertEqual(r["type"], "boolean")

    def test_choice(self):
        r = _question_to_row("s", {"type": "choice", "instructions": "q?",
                                   "criteria": {"a": "da", "b": "db"}})
        self.assertEqual(r["type"], "choice")
        self.assertEqual(r["candidates"], {"a": "da", "b": "db"})

    def test_score(self):
        r = _question_to_row("s", {"type": "score", "instructions": "q?",
                                   "criteria": ["low", "high"]})
        self.assertEqual(r["type"], "score")
        self.assertEqual(r["levels"], ["low", "high"])

    def test_state_passthrough(self):
        r = _question_to_row({"a": 1}, {"type": "boolean", "instructions": "q?"})
        self.assertEqual(r["state"], {"a": 1})


class CardTest(unittest.TestCase):
    def test_card_fills_from_meta(self):
        import json
        import os
        import tempfile

        from haruspex.card import build_card
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"backbone": "Qwen/Qwen3.5-4B", "type": "decoder"}, fh)
        card = build_card(d, "TyrusRC/haruspex")
        self.assertIn("TyrusRC/haruspex", card)
        self.assertIn("Qwen/Qwen3.5-4B", card)
        self.assertIn("/v1/systemone", card)
        self.assertIn("apache-2.0", card)


if __name__ == "__main__":
    unittest.main()
