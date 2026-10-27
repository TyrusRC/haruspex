"""Convert shenjunhao/mmdm -> Haruspex decision-row JSONL, split into train/val.

mmdm (the MMDM benchmark, `test` split, config `default`) rows look like:
  state, question,
  options: [{id, name, criteria}],
  task: "choice" | "score" | "boolean"|"noul",
  score_values: [floats] | None,
  answer: {kind: "hard"|"soft", probabilities: [...]}   # aligned to options order

choice -> SOFT target distribution over option names (kept in the data; gliner2 2.1
          trains on hard labels only, so training uses its argmax).
score  -> level labels + the argmax level index.
boolean/noul -> P(true) from the yes option.

  pip install -e ".[data]"
  python scripts/convert_mmdm.py --out-dir data --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter


def _unique_candidates(opts):
    """{name: criteria} with ids appended on duplicate names; keep option order."""
    cands, keys = {}, []
    for o in opts:
        key = str(o.get("name", ""))
        if key in cands:
            key = f'{key} [{o.get("id", "")}]'
        cands[key] = o.get("criteria") or ""
        keys.append(key)
    return cands, keys


def convert_row(r: dict) -> dict | None:
    task = (r.get("task") or "choice").lower()
    state, question = r.get("state", ""), r.get("question", "")
    opts = r.get("options") or []
    probs = (r.get("answer") or {}).get("probabilities") or []
    if len(opts) < 2:
        return None
    if task == "choice":
        cands, keys = _unique_candidates(opts)
        target = {keys[i]: float(probs[i]) for i in range(min(len(keys), len(probs)))}
        return {"type": "choice", "state": state, "question": question,
                "candidates": cands, "target": target or keys[0]}
    if task == "score":
        levels = [str(o.get("name", i)) for i, o in enumerate(opts)]
        idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
        return {"type": "score", "state": state, "question": question,
                "levels": levels, "target": int(idx)}
    if task in ("boolean", "noul"):
        # map the "yes"/"true" option's probability; fall back to the first option
        names = [str(o.get("name", "")).lower() for o in opts]
        yi = next((i for i, n in enumerate(names) if n in ("yes", "true")), 0)
        p = float(probs[yi]) if yi < len(probs) else 0.0
        return {"type": "boolean", "state": state, "question": question, "target": p}
    return None


def main():
    ap = argparse.ArgumentParser(description="shenjunhao/mmdm -> Haruspex JSONL")
    ap.add_argument("--dataset", default="shenjunhao/mmdm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    rows, types = [], Counter()
    for r in ds:
        row = convert_row(r)
        if row is None:
            continue
        rows.append(row)
        types[row["type"]] += 1
        if args.limit and len(rows) >= args.limit:
            break

    random.Random(args.seed).shuffle(rows)
    cut = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:cut], rows[cut:]
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(f"{args.out_dir}/{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[mmdm] {len(rows)} rows  types={dict(types)}  -> "
          f"{args.out_dir}/train.jsonl ({len(train)}) + val.jsonl ({len(val)})")


if __name__ == "__main__":
    main()
