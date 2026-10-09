"""Decision-row schema, text building, and dataset loading.

On-disk format (JSONL, one decision per line). Three types, mirroring the Jev /
OpenJev contract:

  {"type":"boolean","state":"...","question":"...",
   "criteria":{"true":"...","false":"..."},"target":true}

  {"type":"choice","state":"...","question":"...",
   "candidates":{"billing":"payment problems","shipping":"delivery problems"},
   "target":"billing"}            # or "target":{"billing":0.8,"shipping":0.2}

  {"type":"score","state":"...","question":"...",
   "levels":["low","medium","high"],"target":2}   # index, or a float

Each decision becomes one or more scored inputs (one per candidate / level /
the boolean hypothesis). The encoder scores each input; the readout normalizes
per question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def build_input(state, question: str, candidate: str) -> str:
    """The text scored for one candidate. `state` may be a string or a JSON-able
    object (serialized compactly)."""
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
    cand = candidate.strip() if candidate else ""
    return f"[STATE]\n{s}\n\n[QUESTION]\n{question.strip()}\n\n[CANDIDATE]\n{cand}"


@dataclass
class Example:
    """A normalized, model-ready decision."""

    type: str                 # boolean | choice | score
    texts: list[str]          # one encoded input per candidate (boolean -> 1)
    target: list[float]       # distribution over candidates (boolean -> [p_yes])
    names: list[str] = field(default_factory=list)   # option/level labels for readout


def _choice_target(target, names: list[str]) -> list[float]:
    if isinstance(target, dict):
        return [float(target.get(n, 0.0)) for n in names]
    vec = [0.0] * len(names)
    if target in names:
        vec[names.index(target)] = 1.0
    return vec


def normalize(row: dict) -> Example:
    """Turn one on-disk row into an Example. Raises ValueError on a bad row."""
    t = (row.get("type") or "").lower()
    state, question = row.get("state", ""), row.get("question", "")
    if t == "boolean":
        crit = row.get("criteria") or {}
        hyp = crit.get("true") or "This statement is true."
        y = row.get("target")
        p = float(y) if isinstance(y, (int, float)) and not isinstance(y, bool) else (1.0 if y else 0.0)
        return Example("boolean", [build_input(state, question, hyp)], [p], ["true"])
    if t == "choice":
        cands = row.get("candidates") or {}
        names = list(cands.keys())
        if len(names) < 2:
            raise ValueError("choice needs >=2 candidates")
        texts = [build_input(state, question, f"{n}: {cands[n]}") for n in names]
        return Example("choice", texts, _choice_target(row.get("target"), names), names)
    if t == "score":
        levels = row.get("levels") or []
        if len(levels) < 2:
            raise ValueError("score needs >=2 levels")
        names = [str(x) for x in levels]
        texts = [build_input(state, question, lv) for lv in names]
        tgt = row.get("target")
        vec = [0.0] * len(names)
        if isinstance(tgt, (int, float)) and not isinstance(tgt, bool):
            idx = max(0, min(len(names) - 1, int(round(float(tgt)))))
            vec[idx] = 1.0
        return Example("score", texts, vec, names)
    raise ValueError(f"unknown decision type {t!r}")


def load_jsonl(path: str | Path) -> list[Example]:
    out: list[Example] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(normalize(json.loads(line)))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{i + 1}: {e}") from e
    return out
