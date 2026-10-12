"""Validate a Haruspex decision dataset — enforce "clean" before training.

Checks every row for a well-formed schema, a resolvable label, and no duplicates,
then reports balance by side / type / category. Exits non-zero on any error, so it
doubles as a CI gate.

  python scripts/validate_data.py data/security.jsonl
  python scripts/validate_data.py data/security.jsonl --build   # also build gliner2 examples

Pure Python by default (no torch/gliner2); `--build` additionally confirms each row
turns into a gliner2 training example.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.data import gold_label, labels_ordered, load_rows  # noqa: E402

TYPES = {"boolean", "choice", "score"}


def _row_errors(row: dict, idx: int) -> list[str]:
    errs = []
    t = (row.get("type") or "").lower()
    if t not in TYPES:
        return [f"row {idx}: bad/missing type {row.get('type')!r}"]
    if not str(row.get("state", "")).strip():
        errs.append(f"row {idx}: empty state")
    if not str(row.get("question", "")).strip():
        errs.append(f"row {idx}: empty question")
    if "target" not in row:
        errs.append(f"row {idx}: missing target")
    try:
        labels = labels_ordered(row)
        if len(labels) < 2:
            errs.append(f"row {idx}: <2 labels")
        gold = gold_label(row)
        if gold is None:
            errs.append(f"row {idx}: target does not resolve to a label")
        elif gold not in labels:
            errs.append(f"row {idx}: gold {gold!r} not in labels {labels}")
    except ValueError as e:
        errs.append(f"row {idx}: {e}")
    if t == "boolean":
        crit = row.get("criteria") or {}
        if "true" not in crit or "false" not in crit:
            errs.append(f"row {idx}: boolean needs criteria.true and criteria.false")
    return errs


def validate(path: str, build: bool = False) -> int:
    rows = load_rows(path)
    errors: list[str] = []
    seen: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        errors.extend(_row_errors(row, i))
        key = (row.get("type"), str(row.get("state", "")).strip().lower(),
               str(row.get("question", "")).strip().lower())
        if key in seen:
            errors.append(f"row {i}: duplicate of row {seen[key]} (same type/state/question)")
        else:
            seen[key] = i

    if build and not errors:
        from haruspex.data import to_input_example
        for i, row in enumerate(rows):
            try:
                to_input_example(row)
            except Exception as e:  # noqa: BLE001 - surface any build failure as a data error
                errors.append(f"row {i}: cannot build example: {e}")

    by_type = Counter(r.get("type") for r in rows)
    by_side = Counter(r.get("side", "unspecified") for r in rows)
    by_cat = Counter(r.get("category", "unspecified") for r in rows)
    print(f"[validate] {path}: {len(rows)} rows")
    print(f"  type: {dict(by_type)}")
    print(f"  side: {dict(by_side)}")
    print(f"  category: {dict(sorted(by_cat.items()))}")
    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print("[OK] clean")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Validate a Haruspex decision dataset")
    ap.add_argument("path")
    ap.add_argument("--build", action="store_true", help="also build gliner2 examples (needs gliner2)")
    sys.exit(validate(ap.parse_args().path, ap.parse_args().build))


if __name__ == "__main__":
    main()
