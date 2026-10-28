"""Tests for the novelty gate (ood.py)."""

import numpy as np
import pytest

from haruspex.ood import DIM, NoveltyGate, embed


def test_embed_shape():
    v = embed("test text for embedding")
    assert v.shape == (DIM,)
    assert v.dtype == np.float32


def test_embed_normalized():
    v = embed("any input text")
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5


def test_embed_empty():
    v = embed("")
    assert v.shape == (DIM,)
    # empty text has zero norm — embed returns zero vector
    assert float(np.linalg.norm(v)) < 1e-6


def test_embed_deterministic():
    a = embed("same text twice")
    b = embed("same text twice")
    assert np.allclose(a, b)


def test_embed_different():
    a = embed("SQL injection via UNION SELECT")
    b = embed("the weather is sunny today")
    cosine = float(a @ b)
    assert cosine < 0.9  # different topics should not be near-identical


def test_novelty_gate_in_distribution():
    train = ["SELECT * FROM users", "DROP TABLE students", "INSERT INTO logs VALUES"]
    cal = ["SELECT version()", "UPDATE accounts SET"]
    gate = NoveltyGate.fit(train, cal, alpha=0.05)
    result = gate.check("DELETE FROM orders WHERE id=1")
    assert "novel" in result
    assert "novelty_p" in result
    assert not result["novel"]  # SQL-like text should be in-distribution


def test_novelty_gate_out_of_distribution():
    train = ["SELECT * FROM users", "DROP TABLE students", "INSERT INTO logs VALUES",
             "UPDATE accounts SET balance", "SELECT version()", "UNION SELECT 1,2,3"]
    cal = ["DELETE FROM orders", "ALTER TABLE users ADD COLUMN", "GRANT ALL PRIVILEGES",
           "CREATE INDEX idx ON users", "SELECT COUNT(*) FROM sessions"]
    gate = NoveltyGate.fit(train, cal, alpha=0.3)  # high alpha to catch OOD
    result = gate.check("The mitochondria is the powerhouse of the cell and photosynthesis converts light energy")
    # very different domain — should have low p-value
    assert result["novelty_p"] < 0.5


def test_novelty_gate_save_load(tmp_path):
    train = ["finding one", "finding two", "finding three"]
    cal = ["finding four"]
    gate = NoveltyGate.fit(train, cal)
    gate.save(tmp_path)
    loaded = NoveltyGate.load(tmp_path)
    assert loaded is not None
    assert loaded.alpha == gate.alpha
    assert len(loaded.cal) == len(gate.cal)
    assert loaded.train.shape == gate.train.shape


def test_novelty_gate_load_missing(tmp_path):
    assert NoveltyGate.load(tmp_path) is None


def test_novelty_gate_dim_mismatch(tmp_path):
    train = ["a", "b"]
    gate = NoveltyGate.fit(train, ["c"])
    gate.save(tmp_path)
    import json
    meta = json.loads((tmp_path / "ood.json").read_text())
    meta["dim"] = 999
    (tmp_path / "ood.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="dim"):
        NoveltyGate.load(tmp_path)
