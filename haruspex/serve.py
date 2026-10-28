"""Haruspex decision server — the Jev / TypeSafe contract Praetor / vigil call.

  pip install -e ".[serve]"
  haruspex-serve --model runs/sec --port 3000                       # one adapter
  haruspex-serve --adapters web=runs/web,mobile=runs/mobile --port 3000  # per-domain router
  # or: uvicorn haruspex.serve:app  (set HARUSPEX_MODEL=runs/sec
  #                                  or HARUSPEX_ADAPTERS=web=runs/web,mobile=runs/mobile)

Request  (Jev / TypeSafe shape):
  { "model": "...", "state": <str|obj>, "domain": "web"|"mobile"|...,
    "questions": { "<name>": { "type": "boolean"|"noul"|"choice"|"score",
                               "instructions": "...",
                               "criteria": {true,false} | {opt:desc} | [low,high] } } }
Response:
  { "model": "haruspex", "domain": "<resolved>",
    "answers": { "<name>": { ...calibrated verdict... } } }

One ~0.5B model cannot hold every security domain in a single adapter (cross-domain
interference breaks a merged adapter's clean boundaries). The router keeps a shared
contract behind ONE endpoint and serves the right per-domain adapter per request,
selected by the request's `domain` field; an unknown/missing domain falls back to the
default adapter (fail-open — never an error, never a dropped request).

Self-hosted and keyless — the key authenticates a remote gateway only, so
Praetor/vigil select it with just the base_url (http://127.0.0.1:<port>).
"""

from __future__ import annotations

import argparse
import json
import os

# A `state` beyond this is rejected before it reaches the tokenizer. Measured: a single
# ~2 MB state drove the DeBERTa-v3-large encoder to ~8.5 GB RSS and the process was
# OOM-killed with no error returned to the caller — an unauthenticated request-size DoS.
# 64 KB covers any real finding/code-snippet/IaC-block with headroom; still-oversized
# content should be summarized/truncated by the caller, not sent whole.
MAX_STATE_CHARS = 65536
# The raw body is capped BEFORE it is parsed (the state cap alone runs after json.loads
# has already materialised the whole body), and so is the per-request question count —
# each question is one encoder pass.
MAX_BODY_BYTES = 512 * 1024
MAX_QUESTIONS = 64

# Optional hardening, each off by default (the loopback-trust model stays the default):
#   HARUSPEX_API_KEY=<secret>     require `Authorization: Bearer <secret>` on /v1/evaluate.
#                                  Unset = keyless, as today — only turn this on when the
#                                  server is reachable beyond a single trusted loopback host.
#   HARUSPEX_AUDIT_LOG=<path>     append one JSONL line per request: timestamp, domain,
#                                  routing source, and each question's label/probability/
#                                  source. The raw `state` is NOT logged (it may be
#                                  sensitive finding/code content) — only its length.
import collections
import hmac
import threading
import time

_METRICS_LOCK = threading.Lock()
_METRICS = collections.Counter()

# Keyword signals that infer a domain when a request carries NO explicit `domain`
# (the "smart switch" fallback). Add a domain here as data, not code. Inference runs
# ONLY over registered adapters and ONLY when the caller gave no hint — an explicit
# hint always wins (the tools know their own domain; a content guess can mis-route).
_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "mobile": ("apk", "android", "androidmanifest", "android:exported", "debuggable",
               "strings.xml", "info.plist", "keystore", "webview", "deeplink", "deep link",
               "frida", "smali", ".dex", "minsdk", "sharedpreferences", "app transport security",
               " ats ", ".ipa", "objective-c", "intent filter", "content provider", "ios app"),
    "cloud": ("terraform", "cloudformation", "aws_s3", "s3 bucket", "iam policy", "iam role",
              "security group", "0.0.0.0/0", "publicly_accessible", "boto3", "kms key",
              "cloudtrail", "assume role", "arn:aws", "akia", "block_public_acls", "lambda function",
              "bucket policy", "ec2 instance"),
    "network": ("smb signing", "kerberos", "kerberoast", "as-rep", "netexec", "impacket",
                "bloodhound", "active directory", "domain controller", "ldap bind", "spn",
                "delegation", "llmnr", "nbt-ns", "sysvol", "dcsync", "adcs", "certipy", "ntlm relay"),
    "llm": ("prompt injection", "jailbreak", "system prompt", "llm", "rag ", "excessive agency",
            "prompt leak", "guardrail", "indirect prompt", "system-prompt"),
    "soc": ("defender xdr", "microsoft defender", "sentinel", "evtx", "sysmon", "eventid",
            "security event log", "mitre techniques", "alert categories", "triage grade"),
    "dast": ("dast finding", "scanner finding", "reflected xss", "sql injection detected",
             "command injection", "ssrf", "open redirect", "nuclei template", "vigil finding",
             "false positive", "true positive", "noise", "confidence"),
    "guide": ("template relevant", "wstg", "owasp", "detection rule", "nuclei template",
              "tech stack", "framework", "detector applicable"),
}


def _parse_adapters(spec: str) -> dict[str, str]:
    """Parse `web=dir,mobile=dir2` (or `;`-separated) into {domain: path}.
    A bare path with no `=` registers as the default domain `web`."""
    out: dict[str, str] = {}
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            dom, path = part.split("=", 1)
            out[dom.strip()] = path.strip()
        else:
            out["web"] = part
    return out


class Router:
    """Holds one Haruspex instance per domain behind one endpoint.

    NOTE (ceiling): each domain loads its own base copy (~2 GB) because gliner2's
    load_adapter discards the PEFT wrapper, so clean single-base named-adapter
    switching is not exposed. Fine for a few domains; the upgrade path is PEFT
    named adapters on one shared base once gliner2 exposes set_adapter."""

    def __init__(self, adapters: dict[str, str], default: str = "web"):
        if not adapters:
            raise ValueError("Router needs at least one adapter")
        self.adapters = adapters
        # Default must be a registered domain; fall back to the first registered.
        self.default = default if default in adapters else next(iter(adapters))
        self._cache: dict[str, object] = {}
        self._load_lock = threading.Lock()   # concurrent first requests load ONE copy

    def route(self, domain: str | None, state="") -> tuple[str, str]:
        """Pick the served domain and say how. Returns (domain, source):
          - explicit hint that is registered       -> (hint, "hint")
          - explicit hint that is NOT registered    -> (default, "default")  (respect intent, don't guess)
          - no hint, state matches a domain's signal -> (inferred, "inferred")  (smart switch)
          - no hint, no signal                       -> (default, "default")
        Fail-open: a routing miss costs a less-tuned rank, never an error."""
        d = (domain or "").strip()
        if d in self.adapters:
            return d, "hint"
        if d:
            return self.default, "default"
        inferred = self._infer(state)
        return (inferred, "inferred") if inferred else (self.default, "default")

    def _infer(self, state) -> str | None:
        """Infer a registered domain from the request text, or None. Only a
        non-default domain is worth inferring — the default needs no signal."""
        text = (state if isinstance(state, str) else json.dumps(state, default=str)).lower()
        best, score = None, 0
        for dom, kws in _DOMAIN_HINTS.items():
            if dom not in self.adapters or dom == self.default:
                continue
            c = sum(1 for k in kws if k in text)
            if c > score:
                best, score = dom, c
        return best if score > 0 else None

    def resolve(self, domain: str | None) -> str:
        """Back-compat: the routed domain for a hint only (no state inference)."""
        return self.route(domain, "")[0]

    def model(self, domain: str | None):
        d = self.resolve(domain)
        if d not in self._cache:
            with self._load_lock:
                if d not in self._cache:
                    from .config import pick_device
                    from .model import Haruspex
                    # a spec may name a subfolder in a multi-domain repo: repo#subfolder
                    path, _, sub = self.adapters[d].partition("#")
                    self._cache[d] = Haruspex.load(path, device=pick_device(), subfolder=sub)
        return self._cache[d]


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


def state_len(state) -> int:
    """Size of `state` as the limits count it (a non-string state counts as its JSON)."""
    return len(state) if isinstance(state, str) else len(json.dumps(state, default=str))


def answer_questions(m, state, questions: dict) -> dict:
    """Answer every question about one state; a bad question is a per-question error,
    never a failed request. Valid questions go to the model as ONE batch (`decide_many`)
    when it has one. Shared by the HTTP server and the MCP server."""
    from .data import label_map
    answers, valid = {}, {}
    for name, q in questions.items():
        if not isinstance(q, dict):
            answers[name] = {"error": "question must be an object"}
            continue
        try:
            row = _question_to_row(state, q)
            label_map(row)   # schema check up front, so one bad question cannot sink the batch
            valid[name] = row
        except ValueError as e:
            with _METRICS_LOCK:
                _METRICS["requests_errors"] += 1
            answers[name] = {"error": str(e)}
    if valid:
        many = getattr(m, "decide_many", None)
        verdicts = many(list(valid.values())) if many else [m.decide(r) for r in valid.values()]
        answers.update(zip(valid, verdicts))
    return {name: answers[name] for name in questions}   # keep the caller's order


def build_app(adapters: dict[str, str], default: str = "web"):
    # Built on Starlette directly (FastAPI's base): route handlers receive the
    # request positionally, so there is no body/query heuristic across versions.
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    served_name = os.environ.get("HARUSPEX_SERVED_NAME", "haruspex")
    router = Router(adapters, default=default)
    api_key = os.environ.get("HARUSPEX_API_KEY", "").strip()
    audit_path = os.environ.get("HARUSPEX_AUDIT_LOG", "").strip()
    _audit_lock = threading.Lock()

    def _audit(domain, source, questions, answers, rejected_reason=None):
        if not audit_path:
            return
        rec = {"ts": time.time(), "domain": domain, "domain_source": source,
               "rejected": rejected_reason,
               "answers": {name: {k: v for k, v in (a or {}).items()
                                  if k in ("label", "probability", "source", "error")}
                          for name, a in answers.items()}}
        try:
            with _audit_lock, open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass  # audit logging is best-effort; never fail a request over it

    async def health(request):
        return JSONResponse({"ok": bool(router._cache), "domains": list(router.adapters),
                             "default": router.default})

    async def models(request):
        return JSONResponse({"object": "list",
                             "data": [{"id": served_name, "object": "model",
                                       "domains": list(router.adapters)}]})

    async def metrics(request):
        with _METRICS_LOCK:
            lines = [f"haruspex_{k} {v}" for k, v in sorted(_METRICS.items())]
        from starlette.responses import PlainTextResponse
        return PlainTextResponse("\n".join(lines) + "\n")

    def _bad(status, msg):
        with _METRICS_LOCK:
            _METRICS[f"requests_rejected_{status}"] += 1
        return JSONResponse({"error": msg}, status_code=status)

    async def _read_body(request) -> bytes | None:
        """The raw body, or None once it passes MAX_BODY_BYTES (stops reading there)."""
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_BODY_BYTES:
                return None
        return bytes(buf)

    async def evaluate(request):
        if api_key:
            auth = request.headers.get("authorization", "")
            # constant-time: a plain != leaks the key's matching prefix length via timing
            if not hmac.compare_digest(auth.encode(), f"Bearer {api_key}".encode()):
                with _METRICS_LOCK:
                    _METRICS["requests_unauthorized"] += 1
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        raw = await _read_body(request)
        if raw is None:
            return _bad(413, f"body larger than {MAX_BODY_BYTES} bytes")
        try:
            body = json.loads(raw)
        except ValueError:
            return _bad(400, "body is not valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("questions") or {}, dict):
            return _bad(400, "body must be an object with a 'questions' object")
        state = body.get("state", "")
        n_state = state_len(state)
        questions = body.get("questions") or {}
        if len(questions) > MAX_QUESTIONS:
            return _bad(413, f"too many questions ({len(questions)} > {MAX_QUESTIONS})")
        with _METRICS_LOCK:
            _METRICS["requests_total"] += 1
        if n_state > MAX_STATE_CHARS:
            # Reject before routing/inference touches it — a request this large does not
            # get a domain or a verdict, it gets a clean error for every question asked.
            with _METRICS_LOCK:
                _METRICS["requests_rejected_oversized"] += 1
            err = {"error": f"state too large ({n_state} chars > {MAX_STATE_CHARS} limit)"}
            answers = {name: dict(err) for name in questions}
            _audit(None, "rejected", questions, answers, rejected_reason="oversized_state")
            return JSONResponse({"model": body.get("model") or served_name,
                                 "domain": None, "domain_source": "rejected",
                                 "answers": answers}, status_code=413)
        hint = body.get("domain")
        domain, source = router.route(hint if isinstance(hint, str) else None, state)
        with _METRICS_LOCK:
            _METRICS[f"requests_domain_{domain}"] += 1
        # model loading and inference are blocking: keep them off the event loop so one
        # slow request does not stall every other connection
        m = await run_in_threadpool(router.model, domain)
        answers = await run_in_threadpool(answer_questions, m, state, questions)
        _audit(domain, source, questions, answers)
        return JSONResponse({"model": body.get("model") or served_name,
                             "domain": domain, "domain_source": source, "answers": answers})

    # Warm the default adapter off the main thread so app creation is instant.
    # A first request before it finishes waits on the load lock (same as MCP).
    threading.Thread(target=router.model, args=(router.default,), daemon=True).start()
    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/metrics", metrics, methods=["GET"]),
        # /v1/evaluate = native Jev shape; /v1/systemone = TypeSafe shape; same contract.
        Route("/v1/evaluate", evaluate, methods=["POST"]),
        Route("/v1/systemone", evaluate, methods=["POST"]),
    ])


def _adapters_from_env() -> tuple[dict[str, str], str] | None:
    """HARUSPEX_ADAPTERS (per-domain router) or HARUSPEX_MODEL (single) -> (adapters, default)."""
    spec = os.environ.get("HARUSPEX_ADAPTERS", "").strip()
    if spec:
        return _parse_adapters(spec), os.environ.get("HARUSPEX_DEFAULT_DOMAIN", "web")
    single = os.environ.get("HARUSPEX_MODEL", "").strip()
    if single:
        return {"web": single}, "web"
    return None


# `uvicorn haruspex.serve:app` entry (reads HARUSPEX_ADAPTERS / HARUSPEX_MODEL). Built on
# first access, not at import: importing this module for Router/limits (the MCP server
# does) must not pull in Starlette or load a model.
_app = None


def __getattr__(name):
    global _app
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _app is None:
        env = _adapters_from_env()
        if env is not None:
            _app = build_app(env[0], default=env[1])
    return _app


def main():
    ap = argparse.ArgumentParser(description="Serve Haruspex over /v1/evaluate")
    ap.add_argument("--model", default="", help="single model id/dir (default base); served as domain 'web'")
    ap.add_argument("--adapters", default="",
                    help="per-domain router: web=dir,mobile=dir2 (one /v1/evaluate endpoint, routed by request 'domain')")
    ap.add_argument("--default-domain", default="web", help="domain used when a request omits/mismatches 'domain'")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--ssl-keyfile", default="", help="serve HTTPS: path to a PEM private key (pairs with --ssl-certfile)")
    ap.add_argument("--ssl-certfile", default="", help="serve HTTPS: path to a PEM certificate (pairs with --ssl-keyfile)")
    args = ap.parse_args()
    import uvicorn
    from .config import resolve_model
    if args.adapters:
        adapters = _parse_adapters(args.adapters)
    else:
        adapters = {"web": resolve_model(args.model)}
    tls = {}
    if args.ssl_keyfile or args.ssl_certfile:
        if not (args.ssl_keyfile and args.ssl_certfile):
            raise SystemExit("--ssl-keyfile and --ssl-certfile must be given together")
        tls = {"ssl_keyfile": args.ssl_keyfile, "ssl_certfile": args.ssl_certfile}
    uvicorn.run(build_app(adapters, default=args.default_domain), host=args.host, port=args.port, **tls)


if __name__ == "__main__":
    main()
