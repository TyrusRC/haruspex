"""Deterministic, stratified train/test split for a decision dataset.

Stratifies by (side, category, type) so the held-out set covers every category,
and is seeded so the split is reproducible. The test set is never trained on — it
is the only honest source of a generalization number.

  python scripts/split_data.py data/security.jsonl --test-frac 0.2
  # -> data/security_train.jsonl + data/security_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.data import load_rows  # noqa: E402


def split(rows, test_frac: float, seed: int):
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r.get("side", "?"), r.get("category", "?"), r.get("type", "?"))].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        n_test = max(1, round(len(g) * test_frac)) if len(g) >= 2 else 0
        test.extend(g[:n_test])
        train.extend(g[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    ap = argparse.ArgumentParser(description="Stratified train/test split")
    ap.add_argument("path")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.path)
    train, test = split(rows, args.test_frac, args.seed)
    stem = Path(args.path).with_suffix("")
    for name, part in ((f"{stem}_train.jsonl", train), (f"{stem}_test.jsonl", test)):
        with open(name, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[split] {len(rows)} -> train {len(train)} + test {len(test)}  "
          f"({stem}_train.jsonl, {stem}_test.jsonl)")


if __name__ == "__main__":
    main()
