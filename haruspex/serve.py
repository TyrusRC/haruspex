"""Haruspex /v1/evaluate server — the Jev-compatible contract Praetor / assay call.

  pip install -e ".[serve]"
  haruspex-serve --model runs/haruspex-v0 --port 3000
  # or: uvicorn haruspex.serve:app  (set HARUSPEX_MODEL=runs/haruspex-v0)

Request  (same as hosted Jev):
  { "model": "...", "state": <str|obj>,
    "questions": { "<name>": { "type": "boolean"|"choice"|"score",
                               "instructions": "...",
                               "criteria": {true,false} | {opt:desc} | [low,high] } } }
Response:
  { "model": "haruspex", "answers": { "<name>": { ...calibrated verdict... } } }

It is self-hosted and keyless — the key authenticates a remote gateway only, so
Praetor/assay select it with just the base_url (http://127.0.0.1:<port>).
"""

from __future__ import annotations

import argparse
import os

_MODEL = None


def _load(path: str):
    global _MODEL
    if _MODEL is None:
        from .config import pick_device
        from .model import HaruspexModel
        _MODEL = HaruspexModel.load(path, device=pick_device())
    return _MODEL


def _question_to_row(state, q: dict) -> dict:
    """Map one Jev question into a Haruspex decision row."""
    qt = (q.get("type") or "boolean").lower()
    crit = q.get("criteria")
    row = {"type": qt, "state": state, "question": q.get("instructions", "")}
    if qt == "choice":
        row["candidates"] = crit if isinstance(crit, dict) else {}
    elif qt == "score":
        row["levels"] = crit if isinstance(crit, list) else []
    else:
        row["criteria"] = crit if isinstance(crit, dict) else {}
    return row


def build_app(model_path: str):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Req(BaseModel):
        model: str | None = None
        state: object = ""
        questions: dict = {}

    app = FastAPI(title="haruspex")

    @app.on_event("startup")
    def _startup():
        _load(model_path)

    @app.get("/health")
    def health():
        return {"ok": _MODEL is not None, "model": model_path}

    @app.post("/v1/evaluate")
    def evaluate(req: Req):
        m = _load(model_path)
        answers = {}
        for name, q in (req.questions or {}).items():
            try:
                answers[name] = m.decide(_question_to_row(req.state, q))
            except ValueError as e:
                answers[name] = {"error": str(e)}
        return {"model": req.model or "haruspex", "answers": answers}

    return app


# `uvicorn haruspex.serve:app` entry (reads HARUSPEX_MODEL).
app = None
if os.environ.get("HARUSPEX_MODEL"):
    app = build_app(os.environ["HARUSPEX_MODEL"])


def main():
    ap = argparse.ArgumentParser(description="Serve Haruspex over /v1/evaluate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(build_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
