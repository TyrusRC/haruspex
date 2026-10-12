"""Decision-row helper tests — pure Python, no torch/gliner2, so they run anywhere."""

import os
import tempfile
import unittest

from haruspex.data import (
    decision_text,
    gold_label,
    label_map,
    labels_ordered,
    load_rows,
)


class LabelMapTest(unittest.TestCase):
    def test_boolean(self):
        m = label_map({"type": "boolean", "criteria": {"true": "t", "false": "f"}})
        self.assertEqual(m, {"yes": "t", "no": "f"})

    def test_choice(self):
        m = label_map({"type": "choice", "candidates": {"a": "da", "b": "db"}})
        self.assertEqual(labels_ordered({"type": "choice", "candidates": {"a": "da", "b": "db"}}),
                         ["a", "b"])
        self.assertEqual(m["a"], "da")

    def test_score(self):
        m = label_map({"type": "score", "levels": ["low", "high"]})
        self.assertEqual(m, {"low": "low", "high": "high"})

    def test_choice_needs_two(self):
        with self.assertRaises(ValueError):
            label_map({"type": "choice", "candidates": {"a": ""}})

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            label_map({"type": "teleport"})


class GoldLabelTest(unittest.TestCase):
    def test_boolean_true(self):
        self.assertEqual(gold_label({"type": "boolean", "target": True}), "yes")

    def test_boolean_soft(self):
        self.assertEqual(gold_label({"type": "boolean", "target": 0.3}), "no")

    def test_choice(self):
        self.assertEqual(gold_label({"type": "choice", "candidates": {"a": "", "b": ""},
                                     "target": "b"}), "b")

    def test_choice_soft_dict_argmax(self):
        self.assertEqual(gold_label({"type": "choice", "candidates": {"a": "", "b": ""},
                                     "target": {"a": 0.2, "b": 0.8}}), "b")

    def test_score_index(self):
        self.assertEqual(gold_label({"type": "score", "levels": ["lo", "mid", "hi"],
                                     "target": 2}), "hi")

    def test_unlabeled_is_none(self):
        self.assertIsNone(gold_label({"type": "boolean"}))


class TextAndLoadTest(unittest.TestCase):
    def test_decision_text_joins_state_question(self):
        txt = decision_text({"state": "the state", "question": "the question?"})
        self.assertIn("the state", txt)
        self.assertIn("the question?", txt)

    def test_decision_text_serializes_object_state(self):
        txt = decision_text({"state": {"tech": "nginx"}, "question": "q?"})
        self.assertIn("nginx", txt)

    def test_load_rows_skips_comments_and_blanks(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "rows.jsonl")
        with open(p, "w") as fh:
            fh.write("# comment\n\n")
            fh.write('{"type":"boolean","state":"s","question":"q","target":true}\n')
        rows = load_rows(p)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["type"], "boolean")

    def test_load_rows_bad_json_raises(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "bad.jsonl")
        with open(p, "w") as fh:
            fh.write("{not json}\n")
        with self.assertRaises(ValueError):
            load_rows(p)


if __name__ == "__main__":
    unittest.main()
