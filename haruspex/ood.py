"""Novelty gate: abstain on inputs unlike anything the adapter was trained on.

The calibrator's risk guarantee holds only on data exchangeable with its calibration
split. A log format, product or question style the adapter never saw is exactly where
it fails silently — the encoder still returns a confident label. This gate catches that.

Score: cosine similarity of an input to its NEAREST training row, over hashed character
3/4-gram vectors (stable crc32 hashing, no fitted vocabulary). Conformal novelty test
(Bates et al. 2023): with similarities s_1..s_n of held-out in-distribution cal rows,
an input with similarity s has p = (1 + #{s_i <= s}) / (n + 1), and is novel when
p <= alpha. Guarantee on exchangeable data: P(an in-distribution input is flagged)
<= alpha. Nothing is promised about catching every shift — only about not crying wolf.

NOTE (ceilings + upgrade paths):
- Surface n-grams, not semantics: a paraphrase in a new format may pass, a familiar
  format with new content may not be flagged. Encoder embeddings are the upgrade, once
  gliner2 exposes a pooled representation cheaply.
- Brute-force nearest neighbour over every training row: O(n_train * DIM) per request,
  ~1 ms at a few thousand rows. Use an ANN index past ~100k rows.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

DIM = 2048


def embed(text: str) -> np.ndarray:
    """L2-normalised hashed char 3/4-gram counts."""
    t = " ".join(text.lower().split())
    v = np.zeros(DIM, dtype=np.float32)
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            v[zlib.crc32(t[i:i + n].encode()) % DIM] += 1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


class NoveltyGate:
    def __init__(self, train_vecs: np.ndarray, cal_scores, alpha: float = 0.01):
        self.train = np.asarray(train_vecs, dtype=np.float32)
        self.cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
        self.alpha = alpha

    @classmethod
    def fit(cls, train_texts, cal_texts, alpha: float = 0.01) -> "NoveltyGate":
        train = np.stack([embed(t) for t in train_texts])
        gate = cls(train, [], alpha)
        gate.cal = np.sort([gate.similarity(t) for t in cal_texts])
        return gate

    def similarity(self, text: str) -> float:
        return float((self.train @ embed(text)).max()) if len(self.train) else 0.0

    def p_value(self, text: str) -> float:
        s = self.similarity(text)
        return (1 + int(np.searchsorted(self.cal, s, side="right"))) / (len(self.cal) + 1)

    def check(self, text: str) -> dict:
        p = self.p_value(text)
        return {"novel": p <= self.alpha, "novelty_p": round(p, 6)}

    # ---- persistence: vectors as float16 .npy, scores + alpha as JSON -----------

    def save(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        np.save(d / "ood_train.npy", self.train.astype(np.float16))
        (d / "ood.json").write_text(json.dumps(
            {"alpha": self.alpha, "dim": DIM, "cal_scores": [round(float(x), 6) for x in self.cal]}))

    @classmethod
    def load(cls, model_dir: str | Path) -> "NoveltyGate | None":
        d = Path(model_dir)
        if not (d / "ood.json").exists():
            return None
        meta = json.loads((d / "ood.json").read_text())
        if meta.get("dim") != DIM:
            raise ValueError(f"{d}/ood.json was fitted with dim {meta.get('dim')}, expected {DIM}")
        return cls(np.load(d / "ood_train.npy").astype(np.float32), meta["cal_scores"], meta["alpha"])


def main():
    import argparse

    from .data import decision_text, load_rows

    ap = argparse.ArgumentParser(description="Fit the novelty gate for an adapter dir")
    ap.add_argument("--model", required=True, help="adapter dir to write ood.json / ood_train.npy into")
    ap.add_argument("--train", required=True, help="the adapter's TRAIN split")
    ap.add_argument("--cal", required=True, help="held-out in-distribution split (e.g. the cal split)")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="max rate of flagging in-distribution inputs as novel")
    args = ap.parse_args()
    gate = NoveltyGate.fit([decision_text(r) for r in load_rows(args.train)],
                           [decision_text(r) for r in load_rows(args.cal)], args.alpha)
    gate.save(args.model)
    print(f"[ood] {len(gate.train)} train rows, {len(gate.cal)} cal scores, alpha={args.alpha} "
          f"-> {args.model}/ood.json")


if __name__ == "__main__":
    main()
