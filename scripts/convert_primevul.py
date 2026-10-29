"""Convert PrimeVul (paired) -> Haruspex boolean rows. INDEPENDENT TEST SET ONLY.

PrimeVul is real-world C/C++ functions labelled vulnerable (1) / fixed (0), in matched
pairs (the same function before and after a security patch). It is an independent outside
check for the `code` adapter — nothing like our Juliet training data, and far harder than
a synthetic suite: the vulnerable and fixed members differ only by the fix, so surface
cues don't help.

LICENSE: the upstream PrimeVul licence is not clearly stated (HF mirrors tag MIT, which is
the uploader's choice). Treated as TEST-ONLY and NOT redistributed: this writes a test
file OUTSIDE the repo and refuses an --out inside it, and the rows are never trained on.

The functions are only read as data; nothing is compiled or run. Use `python -I`.

  python -I scripts/convert_primevul.py --src ~/datasets/primevul/data/test.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRITERIA = {"true": "the function contains an exploitable security vulnerability",
            "false": "the function is not vulnerable (the fixed/safe version)"}


def main():
    ap = argparse.ArgumentParser(description="PrimeVul paired -> Haruspex test rows")
    ap.add_argument("--src", default="~/datasets/primevul/data/test.parquet")
    ap.add_argument("--out", default="~/datasets/primevul-rows/primevul_test.jsonl")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="skip functions longer than this (they lose signal once truncated)")
    args = ap.parse_args()
    import pandas as pd
    out = Path(args.out).expanduser().resolve()
    if out.is_relative_to(REPO):
        raise SystemExit(f"[primevul] refusing to write test rows inside the repo: {out}")
    df = pd.read_parquet(Path(args.src).expanduser())
    out.parent.mkdir(parents=True, exist_ok=True)
    n = pos = skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            func = str(r["func"]).strip()
            if not func or len(func) > args.max_chars:
                skipped += 1
                continue
            cwe = r.get("cwe")
            cwe = cwe[0] if isinstance(cwe, (list, tuple)) and len(cwe) else cwe
            q = "Is this function vulnerable?"
            if isinstance(cwe, str) and cwe.startswith("CWE"):
                q = f"Is this function vulnerable ({cwe})?"
            fh.write(json.dumps({"id": f"primevul-{r['idx']}", "side": "defensive",
                                 "category": "code_vuln_real", "source": "primevul",
                                 "type": "boolean", "state": func, "question": q,
                                 "criteria": CRITERIA, "target": bool(int(r["target"]))},
                                ensure_ascii=False) + "\n")
            n += 1
            pos += int(r["target"])
    print(f"[primevul] {n} rows ({pos} vulnerable / {n - pos} fixed), skipped {skipped} "
          f"(empty/too long) -> {out}")


if __name__ == "__main__":
    main()
