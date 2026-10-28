"""Convert NVD CVE records (scripts/fetch_nvd.py output) -> Haruspex score rows.

Each record {id, published, description, severity} becomes one score row: the CVE
description is the state, the NVD-assigned (Primary) CVSS v3 base severity is the target.

Splits are TIME-based for test: every CVE published in the last --test-days of the data
(measured back from the newest `published` in the file) goes to test, so the test set
measures rating CVEs the model has never seen the era of. Older CVEs go to dev / cal /
train by a content hash of the description, so identical (templated) descriptions always
land in the same split; any older description that also appears in test is dropped.
Each split is balanced by capping rows per level. Output is git-ignored (data/*.jsonl).

  python -I scripts/convert_nvd.py --src ~/datasets/nvd/cves.jsonl --out-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = ["low", "medium", "high", "critical"]
QUESTION = "Rate the CVSS v3 base severity."


def _bucket(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def load(path: Path) -> list[dict]:
    """Valid records, first occurrence per CVE id (a fetcher restart can repeat a page)."""
    seen, out = set(), []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                datetime.fromisoformat(r["published"])
            except (ValueError, KeyError, TypeError):
                continue   # a partial last line while the fetcher is writing, or junk
            desc = r.get("description")
            if r.get("severity") in LEVELS and isinstance(desc, str) and desc.strip() \
                    and r.get("id") and r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
    return out


def split(recs: list[dict], test_days: int, seed: int) -> dict[str, list[dict]]:
    parts = {"train": [], "dev": [], "cal": [], "test": []}
    if not recs:
        return parts
    ts = [datetime.fromisoformat(r["published"]) for r in recs]
    cutoff = max(ts) - timedelta(days=test_days)
    test_desc = {r["description"] for r, t in zip(recs, ts) if t >= cutoff}
    for r, t in zip(recs, ts):
        if t >= cutoff:
            parts["test"].append(r)
        elif r["description"] not in test_desc:
            u = _bucket(r["description"], seed)
            parts["dev" if u < 0.15 else "cal" if u < 0.30 else "train"].append(r)
    return parts


def to_row(r: dict, split_name: str) -> dict:
    return {"id": f"nvd-{split_name}-{r['id']}", "side": "defensive",
            "category": "nvd_cvss_severity", "source": "nvd", "type": "score",
            "state": r["description"], "question": QUESTION, "levels": LEVELS,
            "target": LEVELS.index(r["severity"])}


def balanced(recs: list[dict], per_level: int, seed: int, split_name: str) -> list[dict]:
    by: dict[str, list] = {lv: [] for lv in LEVELS}
    for r in sorted(recs, key=lambda r: r["id"]):
        by[r["severity"]].append(r)
    rng, out = random.Random(seed), []
    for rs in by.values():
        rng.shuffle(rs)
        out += [to_row(r, split_name) for r in rs[:per_level]]
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="NVD CVEs -> Haruspex CVSS-severity rows")
    ap.add_argument("--src", default="~/datasets/nvd/cves.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--test-days", type=int, default=365, help="newest N days of data -> test")
    ap.add_argument("--per-level", type=int, default=500, help="train rows per level")
    ap.add_argument("--eval-per-level", type=int, default=150, help="dev / cal rows per level")
    ap.add_argument("--test-per-level", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = load(Path(args.src).expanduser())
    parts = split(recs, args.test_days, args.seed)
    print(f"[nvd] {len(recs)} records; available per split: "
          + ", ".join(f"{s}={dict(Counter(r['severity'] for r in v))}" for s, v in parts.items()))
    caps = {"train": args.per_level, "dev": args.eval_per_level, "cal": args.eval_per_level,
            "test": args.test_per_level}
    for s, cap in caps.items():
        rows = balanced(parts[s], cap, args.seed, s)
        path = out / f"nvd_{s}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[nvd] {s}: {len(rows)} rows "
              f"{dict(Counter(LEVELS[r['target']] for r in rows))} -> {path}")


if __name__ == "__main__":
    main()
