"""Pull a public decision dataset from the HF Hub and convert it to Haruspex
decision-row JSONL (data/schema.md).

General decision data (not security-only) is what lifts JevBench "Intelligence".
Good sources: shenjunhao/mmdm, and any OpenJevData release. Schemas vary, so this
prints the detected columns and uses a configurable field mapping — adjust the
--map flags to the dataset you pull.

  pip install -e ".[data]"
  python scripts/fetch_data.py --dataset shenjunhao/mmdm --split train --out data/train.jsonl
  # inspect the printed columns, then re-run with mappings if needed:
  python scripts/fetch_data.py --dataset <id> --split train --out data/train.jsonl \
      --state-col context --question-col question --options-col options --target-col answer
"""

from __future__ import annotations

import argparse
import json


def _row_to_decision(r: dict, m: dict) -> dict | None:
    state = r.get(m["state"], "")
    question = r.get(m["question"], "")
    opts = r.get(m["options"])
    target = r.get(m["target"])
    if isinstance(opts, dict) and len(opts) >= 2:
        return {"type": "choice", "state": state, "question": question,
                "candidates": {str(k): str(v) for k, v in opts.items()}, "target": target}
    if isinstance(opts, list) and len(opts) >= 2:
        cands = {str(o): "" for o in opts}
        return {"type": "choice", "state": state, "question": question,
                "candidates": cands, "target": str(target)}
    if isinstance(target, bool) or str(target).lower() in ("true", "false", "yes", "no"):
        y = str(target).lower() in ("true", "yes")
        return {"type": "boolean", "state": state, "question": question, "target": y}
    return None


def main():
    ap = argparse.ArgumentParser(description="HF decision dataset -> Haruspex JSONL")
    ap.add_argument("--dataset", required=True, help="HF dataset id (e.g. shenjunhao/mmdm)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--state-col", default="state")
    ap.add_argument("--question-col", default="question")
    ap.add_argument("--options-col", default="candidates")
    ap.add_argument("--target-col", default="target")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    print(f"[fetch] {args.dataset}:{args.split}  rows={len(ds)}  columns={ds.column_names}")

    mp = {"state": args.state_col, "question": args.question_col,
          "options": args.options_col, "target": args.target_col}
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in ds:
            row = _row_to_decision(r, mp)
            if row is None:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"[fetch] wrote {n} decision rows -> {args.out}"
          + ("" if n else "  (0 — adjust the --*-col mappings to the printed columns)"))


if __name__ == "__main__":
    main()
