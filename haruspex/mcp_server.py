"""Haruspex as an MCP tool server (stdio) for AI agents.

  pip install -e ".[mcp]"
  haruspex-mcp --adapters web=runs/web,mobile=runs/mobile   # or HARUSPEX_ADAPTERS / HARUSPEX_MODEL

Tools:
  decide(state, question, type, criteria|candidates|levels, domain?) -> calibrated verdict
  list_domains()                                                    -> served domains + routing

Routing, request limits and the question->row mapping are serve.py's (one code path).
The raw `state` is never logged. stdout is the MCP channel: nothing else writes to it.
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .serve import (MAX_BODY_BYTES, MAX_STATE_CHARS, _DOMAIN_HINTS, Router,
                    _adapters_from_env, _parse_adapters, answer_questions, state_len)

ESCALATE = ("ABSTAINED: Haruspex is not confident enough to decide this within its "
            "calibrated risk budget. Do NOT act on this verdict - escalate to a stronger "
            "model or a human reviewer.")
ACCEPTED = "ACCEPTED: verdict is within the calibrated risk budget; safe to act on."
UNCALIBRATED = ("UNCALIBRATED: this adapter has no calibrator, so there is no abstain "
                "gate. Treat the verdict as advisory and escalate anything consequential.")

_CRITERIA_FIELD = {"boolean": "criteria", "choice": "candidates", "score": "levels"}


class DecideError(ValueError):
    """A request the tool refuses (bad input or over a limit). Message is safe to return."""


def make_tools(router: Router):
    """The tool functions bound to one router (plain callables: testable without MCP)."""

    def decide(state: str | dict[str, Any], question: str,
               type: Literal["boolean", "choice", "score"] = "boolean",
               criteria: dict[str, str] | None = None,
               candidates: dict[str, str] | None = None,
               levels: list[str] | None = None,
               domain: str | None = None) -> dict[str, Any]:
        """Ask Haruspex, a small local calibrated security decision model, one question
        about a piece of security evidence (finding, alert, code snippet, IaC block).

        - type="boolean": yes/no; optional `criteria` = {"true": "...", "false": "..."}.
        - type="choice": pick one of `candidates` = {"option": "description", ...}.
        - type="score": pick one of ordered `levels` = ["low", ..., "high"].
        - domain: optional adapter hint (see list_domains); omitted = inferred from state.

        The result always carries `abstain` and `action`. When abstain is true the model
        declined: do not act on the label - escalate to a stronger model or a human.
        """
        try:
            t = (type or "boolean").lower()
            if t not in _CRITERIA_FIELD:
                raise DecideError(f"type must be one of {sorted(_CRITERIA_FIELD)}")
            if not isinstance(question, str) or not question.strip():
                raise DecideError("question must be a non-empty string")
            n = state_len(state)
            if n > MAX_STATE_CHARS:
                raise DecideError(f"state too large ({n} chars > {MAX_STATE_CHARS} limit); "
                                  "summarize or truncate it and retry")
            crit = {"boolean": criteria, "choice": candidates, "score": levels}[t]
            if t != "boolean" and not crit:
                raise DecideError(f"type={t} requires `{_CRITERIA_FIELD[t]}`")
            # the same overall cap the HTTP body has (question + criteria are unbounded otherwise)
            size = len(json.dumps([question, crit, domain], default=str)) + n
            if size > MAX_BODY_BYTES:
                raise DecideError(f"request larger than {MAX_BODY_BYTES} bytes")
            routed, source = router.route(domain if isinstance(domain, str) else None, state)
            m = router.model(routed)
            q = {"type": t, "instructions": question, "criteria": crit}
            verdict = answer_questions(m, state, {"decision": q})["decision"]
            if "error" in verdict:
                raise DecideError(verdict["error"])
        except DecideError as e:
            raise ToolError(str(e)) from None
        return _with_action(dict(verdict), routed, source)

    def list_domains() -> dict[str, Any]:
        """List the security domains Haruspex serves (one adapter each), the default
        used when `domain` is omitted or unknown, and which domains are auto-inferred
        from state keywords when no `domain` is given."""
        return {"domains": list(router.adapters), "default": router.default,
                "inferred_without_hint": [d for d in _DOMAIN_HINTS
                                          if d in router.adapters and d != router.default],
                "routing": "explicit registered domain wins; unknown domain -> default; "
                           "no domain -> keyword inference over state, else default"}

    return decide, list_domains


def _with_action(v: dict, domain: str, source: str) -> dict:
    """Attach the routed domain and an explicit, agent-actionable next step."""
    v["domain"], v["domain_source"] = domain, source
    if v.get("abstain"):
        msg = ESCALATE
        if v.get("reason"):
            msg += f" Reason: {v['reason']}."
        if v.get("route"):
            msg += f" Suggested route: {v['route']}."
        v["escalate"], v["action"] = True, msg
    elif v.get("accept"):
        v["escalate"], v["action"] = False, ACCEPTED
    else:
        v["abstain"] = v.get("abstain", False)
        v["escalate"], v["action"] = False, UNCALIBRATED
    return v


def build_server(router: Router) -> MCPServer:
    server = MCPServer("haruspex", instructions=(
        "Calibrated security decisions (boolean/choice/score) from a small local model. "
        "Always check `abstain`: when true, escalate to a stronger model or a human."))
    decide, list_domains = make_tools(router)
    server.tool()(decide)
    server.tool()(list_domains)
    return server


def main():
    ap = argparse.ArgumentParser(description="Serve Haruspex as an MCP stdio tool server")
    ap.add_argument("--model", default="", help="single model id/dir; served as domain 'web'")
    ap.add_argument("--adapters", default="", help="per-domain router: web=dir,mobile=dir2")
    ap.add_argument("--default-domain", default="", help="domain used when a call omits/mismatches 'domain'")
    args = ap.parse_args()
    from .config import resolve_model
    env = _adapters_from_env()
    if args.adapters:
        adapters, default = _parse_adapters(args.adapters), "web"
    elif args.model or env is None:
        adapters, default = {"web": resolve_model(args.model)}, "web"
    else:
        adapters, default = env
    router = Router(adapters, default=args.default_domain or default)
    # Warm the default adapter off the stdio loop so the MCP handshake is not held up by
    # a multi-second model load; a first call before it finishes waits on the load lock.
    threading.Thread(target=router.model, args=(router.default,), daemon=True).start()
    build_server(router).run()   # stdio


if __name__ == "__main__":
    main()
