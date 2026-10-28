"""Haruspex demo Space: ask one bounded security decision, get a calibrated verdict.

Runs on a free CPU Space. Adapters load lazily from the TyrusRC/haruspex model repo and
only the most recently used ones stay in memory (each is ~2 GB on CPU).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import gradio as gr

from haruspex import Haruspex
from haruspex.data import label_map

REPO = "TyrusRC/haruspex"
DOMAINS = {
    "web": "", "soc": "soc", "mobile": "mobile", "cloud": "cloud",
    "network": "network", "llm": "llm", "general": "general",
}
MAX_STATE = 64 * 1024       # same cap as the HTTP server
MAX_OPTIONS = 32
KEEP = 2                    # adapters held in memory at once (free CPU Space: 16 GB)

_models: "OrderedDict[str, Haruspex]" = OrderedDict()
_lock = threading.Lock()


def _model(domain: str) -> Haruspex:
    with _lock:
        if domain in _models:
            _models.move_to_end(domain)
            return _models[domain]
        while len(_models) >= KEEP:
            _models.popitem(last=False)
        m = Haruspex.load(REPO, device="cpu", subfolder=DOMAINS[domain])
        _models[domain] = m
        return m


def _options(kind: str, text: str):
    """Parse the options box: boolean 'true: ...' / 'false: ...' lines; choice
    'label: description' lines; score one level per line, lowest first."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) > MAX_OPTIONS:
        raise gr.Error(f"at most {MAX_OPTIONS} options")
    if kind == "score":
        return {"levels": lines}
    pairs = {}
    for ln in lines:
        name, _, desc = ln.partition(":")
        pairs[name.strip()] = desc.strip() or name.strip()
    if kind == "boolean":
        return {"criteria": {"true": pairs.get("true", "yes"), "false": pairs.get("false", "no")}}
    return {"candidates": pairs}


def decide(domain, kind, state, question, options):
    if domain not in DOMAINS:
        raise gr.Error("unknown domain")
    if kind not in ("boolean", "choice", "score"):
        raise gr.Error("unknown decision type")
    state = (state or "").strip()
    if not state or not (question or "").strip():
        raise gr.Error("state and question are required")
    if len(state) > MAX_STATE:
        raise gr.Error(f"state is over {MAX_STATE} characters; summarise it first")
    try:   # raw log JSON (Defender, Sentinel, ECS, Windows events) is normalised by the model
        parsed = json.loads(state)
        if isinstance(parsed, (dict, list)):
            state = parsed
    except ValueError:
        pass
    row = {"type": kind, "state": state, "question": question.strip(), **_options(kind, options)}
    try:
        label_map(row)
    except ValueError as e:
        raise gr.Error(str(e)) from None
    v = _model(domain).decide(row)
    answer = v.get("label") or v.get("choice") or v.get("level")
    if v.get("abstain"):
        head = f"**Abstained**: {v.get('reason') or 'not confident enough for the risk budget'}. Escalate to a stronger model or a human."
        if answer:
            head += f"\n\nBest guess: `{answer}` (confidence {v.get('confidence')})"
    else:
        head = f"**{answer}** (confidence {v.get('confidence')}, source `{v.get('source')}`)"
    if v.get("prediction_set"):
        head += f"\n\nPrediction set: {', '.join(f'`{x}`' for x in v['prediction_set'])}"
    return head, v


# Real held-out test rows the published adapters answer correctly (not cherry-picked prompts).
with open(Path(__file__).with_name("examples.json")) as f:
    EXAMPLES = json.load(f)

with gr.Blocks(title="Haruspex") as demo:
    gr.Markdown(
        "# Haruspex\nA small, local, calibrated decision layer for security operations. "
        "Pick a domain, describe the situation, ask a bounded question. Haruspex returns a "
        "typed verdict with its full probability distribution, and **abstains** when it is "
        "not sure enough. [Model](https://huggingface.co/TyrusRC/haruspex) · "
        "[Code](https://github.com/TyrusRC/haruspex)\n\n"
        "The first request per domain downloads and loads its adapter, which takes a minute.")
    with gr.Row():
        domain = gr.Dropdown(list(DOMAINS), value="web", label="Domain")
        kind = gr.Dropdown(["boolean", "choice", "score"], value="boolean", label="Decision type")
    state = gr.Textbox(lines=6, label="State (text, or raw Defender / Sentinel / ECS / Windows event JSON)")
    question = gr.Textbox(label="Question")
    options = gr.Textbox(lines=4, label="Options: boolean 'true: ...' / 'false: ...'; "
                                        "choice 'label: description' per line; score one level per line")
    go = gr.Button("Decide", variant="primary")
    summary = gr.Markdown()
    verdict = gr.JSON(label="Verdict")
    go.click(decide, [domain, kind, state, question, options], [summary, verdict],
             concurrency_limit=1)
    gr.Examples(EXAMPLES, [domain, kind, state, question, options])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
