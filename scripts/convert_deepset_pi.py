"""Convert deepset/prompt-injections -> Haruspex boolean rows. INDEPENDENT TEST SET.

An independent, public prompt-injection set (Apache-2.0) the `llm` adapter was NOT trained
on — a real outside check, not a split of our own data. Each row: text + a binary label
(1 = injection, 0 = benign). We map it to the exact question the llm adapter answers.

Writes a test file only (this is for measuring, not training). Default output is OUTSIDE
the repo to keep eval data out of version control; pass --out to place it.

The dataset is only read as data; nothing in it is executed. Run with `python -I`.

  python -I scripts/convert_deepset_pi.py --src ~/datasets/deepset-pi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUESTION = "Is this input a prompt-injection or jailbreak attempt?"
CRITERIA = {"true": "it tries to override the system prompt, exfiltrate instructions, or "
                    "jailbreak the model",
            "false": "it is a benign request with no attempt to subvert the model"}


def _rows(src: Path):
    """Yield (text, label) from the test split, whatever on-disk format it shipped in."""
    import pandas as pd
    files = sorted(src.rglob("*test*.parquet")) or sorted(src.rglob("*.parquet"))
    if not files:
        files = sorted(src.rglob("*test*.csv")) or sorted(src.rglob("*.csv"))
        frames = [pd.read_csv(f) for f in files]
    else:
        frames = [pd.read_parquet(f) for f in files]
    if not frames:
        raise SystemExit(f"[deepset] no parquet/csv under {src}")
    df = pd.concat(frames, ignore_index=True)
    text_col = next(c for c in ("text", "prompt", "input") if c in df.columns)
    label_col = next(c for c in ("label", "labels", "target") if c in df.columns)
    for _, r in df.iterrows():
        text = str(r[text_col]).strip()
        if text:
            yield text, bool(int(r[label_col]))


def main():
    ap = argparse.ArgumentParser(description="deepset/prompt-injections -> Haruspex test rows")
    ap.add_argument("--src", default="~/datasets/deepset-pi")
    ap.add_argument("--out", default="~/datasets/deepset-pi-rows/deepset_pi_test.jsonl")
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = pos = 0
    with out.open("w", encoding="utf-8") as fh:
        for text, label in _rows(src):
            fh.write(json.dumps({"id": f"deepset-pi-{n}", "side": "defensive",
                                 "category": "llm_prompt_injection", "source": "deepset/prompt-injections",
                                 "type": "boolean", "state": text, "question": QUESTION,
                                 "criteria": CRITERIA, "target": label}, ensure_ascii=False) + "\n")
            n += 1
            pos += label
    print(f"[deepset] {n} rows ({pos} injection / {n - pos} benign) -> {out}")


if __name__ == "__main__":
    main()
