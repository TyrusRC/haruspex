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
    """Map one question (Jev or TypeSafe shape) into a Haruspex decision row.
    TypeSafe calls boolean 'noul'."""
    qt = (q.get("type") or "boolean").lower()
    crit = q.get("criteria")
    if qt == "noul":
        qt = "boolean"
    row = {"type": qt, "state": state, "question": q.get("instructions", "")}
    if qt == "choice":
        row["candidates"] = crit if isinstance(crit, dict) else {}
    elif qt == "score":
        row["levels"] = crit if isinstance(crit, list) else []
    else:
        row["criteria"] = crit if isinstance(crit, dict) else {}
    return row


def build_app(model_path: str):
    # Built on Starlette directly (FastAPI's base): route handlers receive the
    # request positionally, so there is no body/query parameter heuristic to
    # trip over across versions. `state` may be a string OR a JSON object.
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    served_name = os.environ.get("HARUSPEX_SERVED_NAME", "haruspex")

    async def health(request):
        return JSONResponse({"ok": _MODEL is not None, "model": model_path})

    async def models(request):
        # OpenAI-style model list — the JevBench adapter / TypeSafe clients probe this.
        return JSONResponse({"object": "list",
                             "data": [{"id": served_name, "object": "model"}]})

    async def evaluate(request):
        m = _load(model_path)
        body = await request.json()
        state = body.get("state", "")
        answers = {}
        for name, q in (body.get("questions") or {}).items():
            try:
                answers[name] = m.decide(_question_to_row(state, q))
            except ValueError as e:
                answers[name] = {"error": str(e)}
        return JSONResponse({"model": body.get("model") or served_name, "answers": answers})

    _load(model_path)   # warm the model before serving (version-proof: no lifespan hook)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            # /v1/evaluate = native Jev (Praetor/assay); /v1/systemone = TypeSafe shape
            # (JevBench's open-weights adapter). Same request/response contract.
            Route("/v1/evaluate", evaluate, methods=["POST"]),
            Route("/v1/systemone", evaluate, methods=["POST"]),
        ],
    )


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
