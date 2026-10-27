"""Deterministic, stable train/dev/cal/test split for a decision dataset.

Each row's split is a hash of its own content (and the seed), so the split is
reproducible AND stable as the dataset grows. The test set is never trained on — it
is the only honest source of a generalization number.

Four roles, each used for exactly one thing:
  train  fit the adapter
  dev    choose epochs / rank / seed (and fit temperature)
  cal    fit the selective calibrator — never used for tuning, or its guarantee breaks
  test   report the final number, once

  python scripts/split_data.py data/security.jsonl --test-frac 0.2 --cal-frac 0.15 --dev-frac 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.data import load_rows  # noqa: E402


def _unit(row: dict, seed: int) -> float:
    """A stable uniform draw in [0, 1) from the row's own content: the same row lands in
    the same split every time, however many rows are added around it."""
    key = json.dumps([seed, row.get("type"), row.get("state"), row.get("question")],
                     sort_keys=True, ensure_ascii=False, default=str)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2 ** 64


def split(rows, test_frac: float, seed: int, cal_frac: float = 0.0, dev_frac: float = 0.0):
    """Content-hashed split into train/cal/test/dev (cal and dev empty when their
    fraction is 0). Returns (train, cal, test, dev).

    Membership depends only on the row and the seed, never on the rest of the file:
    appending rows can never move a row an adapter trained on into test (the earlier
    shuffle-based split did exactly that). Category balance holds in expectation.
    NOTE (ceiling): a category with a handful of rows may get no test row; check the
    printed per-split counts, and add rows to that category rather than reshuffling."""
    cuts = (test_frac, test_frac + cal_frac, test_frac + cal_frac + dev_frac)
    train, cal, test, dev = [], [], [], []
    for r in rows:
        u = _unit(r, seed)
        (test if u < cuts[0] else cal if u < cuts[1] else dev if u < cuts[2] else train).append(r)
    return train, cal, test, dev


def main():
    ap = argparse.ArgumentParser(description="Stratified train/(cal)/test split")
    ap.add_argument("path")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cal-frac", type=float, default=0.0,
                    help="held-out calibration fraction (>0 writes a *_cal.jsonl)")
    ap.add_argument("--dev-frac", type=float, default=0.0,
                    help="held-out dev fraction for choosing hyperparameters (>0 writes a *_dev.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.path)
    train, cal, test, dev = split(rows, args.test_frac, args.seed, args.cal_frac, args.dev_frac)
    stem = Path(args.path).with_suffix("")
    parts = [(f"{stem}_train.jsonl", train), (f"{stem}_test.jsonl", test)]
    if cal:
        parts.append((f"{stem}_cal.jsonl", cal))
    if dev:
        parts.append((f"{stem}_dev.jsonl", dev))
    for name, part in parts:
        with open(name, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tail = (f" + cal {len(cal)}" if cal else "") + (f" + dev {len(dev)}" if dev else "")
    print(f"[split] {len(rows)} -> train {len(train)}{tail} + test {len(test)}")
    tested = Counter(r.get("category", "?") for r in test)
    missing = sorted({r.get("category", "?") for r in rows} - set(tested))
    if missing:
        print(f"[split] WARNING: no test rows for categories: {missing}")


if __name__ == "__main__":
    main()
