"""Head-to-head benchmark: Haruspex vs. any OpenAI-compatible LLM on the same rows.

The LLM gets a fair shot: the same state/question/option descriptions Haruspex sees,
greedy decoding, output constrained to the valid labels (JSON schema enum), and
optionally k same-category worked examples drawn from the TRAIN split only.

  llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 8192 --port 8088
  python scripts/bench.py --data data/security_test.jsonl --model runs/sec-v2 \\
      --llm-url http://127.0.0.1:8088 --shots 3 --shots-from data/security_train.jsonl

Prints each side's metrics, a per-category table, and the exact McNemar p-value.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.data import gold_label, label_map, load_rows  # noqa: E402
from haruspex.eval import _mcnemar_p, score_rows  # noqa: E402

_SYSTEM = ("You are a senior security engineer making a bounded decision. Read the "
           "situation and pick exactly one option. Answer with JSON only.")


def _state(row) -> str:
    s = row.get("state", "")
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def _prompt(row) -> str:
    opts = "\n".join(f"- {k}: {v}" for k, v in label_map(row).items())
    order = " (ordered lowest to highest)" if row.get("type") == "score" else ""
    return (f"Situation:\n{_state(row)}\n\nQuestion: {row.get('question', '')}\n\n"
            f"Options{order}:\n{opts}\n\nReply as {{\"answer\": \"<option>\"}}.")


def _shots(row, pool, k):
    same = [r for r in pool if r.get("category") == row.get("category") and r is not row]
    msgs = []
    for ex in same[:k]:
        msgs += [{"role": "user", "content": _prompt(ex)},
                 {"role": "assistant", "content": json.dumps({"answer": gold_label(ex)})}]
    return msgs


def _llm_fn(url: str, pool, k: int, timeout: float = 120):
    def fn(row):
        labels = list(label_map(row))
        body = {"messages": [{"role": "system", "content": _SYSTEM}, *_shots(row, pool, k),
                             {"role": "user", "content": _prompt(row)}],
                "temperature": 0, "max_tokens": 40,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {
                    "type": "object", "properties": {"answer": {"type": "string", "enum": labels}},
                    "required": ["answer"]}}}}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # local server only
            text = json.load(r)["choices"][0]["message"]["content"]
        try:
            ans = json.loads(text).get("answer")
        except (ValueError, AttributeError):
            ans = next((lb for lb in labels if lb.lower() in text.lower()), None)
        t = row.get("type", "boolean")
        key = {"boolean": "label", "choice": "choice", "score": "level"}[t]
        return {"type": t, key: ans, "confidence": 1.0, "source": "llm"}
    return fn


def _by_category(rows, oks):
    acc = defaultdict(list)
    for r, ok in zip([r for r in rows if gold_label(r) is not None], oks):
        acc[r.get("category", "?")].append(ok)
    return {c: (sum(v) / len(v), len(v)) for c, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser(description="Haruspex vs. an OpenAI-compatible LLM")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="Haruspex model dir / id")
    ap.add_argument("--llm-url", required=True, help="OpenAI-compatible base URL (local)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--shots-from", default="", help="train split to draw examples from")
    ap.add_argument("--out", default="", help="write per-row results JSON here")
    args = ap.parse_args()
    if args.shots and not args.shots_from:
        sys.exit("--shots needs --shots-from (a TRAIN split, never the eval data)")

    rows = load_rows(args.data)
    pool = load_rows(args.shots_from) if args.shots_from else []

    from haruspex.model import Haruspex
    hx = Haruspex.load(args.model)
    ours, theirs = [], []
    a = score_rows(hx.decide, rows, ours)
    b = score_rows(_llm_fn(args.llm_url, pool, args.shots), rows, theirs)
    print("haruspex:", json.dumps(a))
    print("llm     :", json.dumps(b))
    print(f"mcnemar : p={_mcnemar_p(ours, theirs):.4g}  (paired exact; < 0.05 = a real difference)")
    ca, cb = _by_category(rows, ours), _by_category(rows, theirs)
    print(f"\n{'category':28} {'n':>3} {'haruspex':>9} {'llm':>6}")
    for c in ca:
        print(f"{c:28} {ca[c][1]:>3} {ca[c][0]:>9.2f} {cb[c][0]:>6.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"haruspex": a, "llm": b, "ours": ours,
                                              "theirs": theirs}, indent=1))


if __name__ == "__main__":
    main()
