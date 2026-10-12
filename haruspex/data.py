"""Decision-row schema + helpers.

On-disk format (JSONL, one decision per line), unchanged across the rebuild:

  {"type":"boolean","state":"...","question":"...",
   "criteria":{"true":"...","false":"..."},"target":true}

  {"type":"choice","state":"...","question":"...",
   "candidates":{"bfla":"broken function-level auth","tls":"TLS config"},
   "target":"bfla"}

  {"type":"score","state":"...","question":"...",
   "levels":["low","medium","high","critical"],"target":3}   # index into levels

The model layer maps a row to a single GLiNER2 classification task (the task
labels are the candidates / levels / yes-no). These helpers are pure Python so
they import without torch/gliner2.
"""

from __future__ import annotations

import json
from pathlib import Path


def decision_text(row: dict) -> str:
    """The text the model classifies: the state followed by the question.
    `state` may be a string or a JSON-able object (serialized compactly)."""
    state = row.get("state", "")
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
    return f"{s} {row.get('question', '')}".strip()


def label_map(row: dict) -> dict[str, str]:
    """Ordered {label: description} for the row's classification task.
    boolean -> yes/no with the criteria as descriptions; choice -> candidates;
    score -> levels (the level name is its own description)."""
    t = (row.get("type") or "").lower()
    if t == "boolean":
        crit = row.get("criteria") or {}
        return {"yes": crit.get("true", "yes"), "no": crit.get("false", "no")}
    if t == "choice":
        cands = dict(row.get("candidates") or {})
        if len(cands) < 2:
            raise ValueError("choice needs >=2 candidates")
        return cands
    if t == "score":
        levels = [str(x) for x in (row.get("levels") or [])]
        if len(levels) < 2:
            raise ValueError("score needs >=2 levels")
        return {lv: lv for lv in levels}
    raise ValueError(f"unknown decision type {t!r}")


def labels_ordered(row: dict) -> list[str]:
    return list(label_map(row).keys())


def gold_label(row: dict):
    """The correct label string for a row, or None when the row is unlabeled."""
    t = (row.get("type") or "").lower()
    tgt = row.get("target")
    if tgt is None:
        return None
    if t == "boolean":
        return "yes" if (tgt is True or (isinstance(tgt, (int, float)) and tgt >= 0.5)) else "no"
    if t == "choice":
        if isinstance(tgt, dict):                 # soft distribution -> argmax label
            return max(tgt, key=tgt.get) if tgt else None
        return str(tgt)
    if t == "score":
        levels = [str(x) for x in (row.get("levels") or [])]
        idx = max(0, min(len(levels) - 1, int(round(float(tgt)))))
        return levels[idx] if levels else None
    raise ValueError(f"unknown decision type {t!r}")


def load_rows(path: str | Path) -> list[dict]:
    """Read decision rows from JSONL (blank lines and # comments skipped)."""
    out: list[dict] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i + 1}: {e}") from e
    return out


def to_input_example(row: dict):
    """One labeled row -> a GLiNER2 `InputExample` with a single classification
    task. Imports gliner2 lazily (training-only dependency)."""
    from gliner2 import InputExample
    from gliner2.training import Classification

    descs = label_map(row)
    gold = gold_label(row)
    if gold is None:
        raise ValueError("training row needs a 'target'")
    cls = Classification("decision", list(descs.keys()), gold, label_descriptions=descs)
    return InputExample(text=decision_text(row), classifications=[cls])
