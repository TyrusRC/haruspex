# Deploying Haruspex beyond a single trusted loopback host

The default deployment — one `haruspex-serve` process, keyless, on `127.0.0.1` — is
correct and sufficient for its primary case: one tool on one box calling its own local
triage layer. Everything below is for when that stops being true: the server is
reachable from another host, from more than one caller, or needs to survive a crash
unattended.

## Auth: `HARUSPEX_API_KEY`

Unset by default (keyless). When the server is reachable beyond one trusted loopback
process, set it:

```sh
HARUSPEX_API_KEY=<secret> haruspex-serve --adapters ...
```

Every `/v1/evaluate` / `/v1/systemone` request then needs `Authorization: Bearer
<secret>`, or it gets a `401`. This is a shared secret, not per-caller auth — fine for
"one trusted network, several callers," not a substitute for a real multi-tenant auth
layer if callers should not see each other's traffic.

## TLS

`uvicorn` serves TLS natively — no reverse proxy required for a simple deployment:

```sh
haruspex-serve --adapters ... --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Verified: HTTPS requests succeed, plain HTTP on the same port is rejected (connection
reset, not a silent plaintext fallback). For anything beyond a single process — load
balancing, cert rotation via ACME, HTTP/2 — terminate TLS at a reverse proxy (nginx,
Caddy, an internal ALB) in front of a plain-HTTP `haruspex-serve` instead; both are
legitimate, pick the one that matches the rest of your fleet's pattern.

## Process supervision

`deploy/haruspex.service` is a systemd unit: `Restart=on-failure` with a capped retry
rate, a memory ceiling sized for N adapters in RAM (~2 GB each on CPU), and least-
privilege sandboxing (`ProtectSystem=strict`, `PrivateTmp`, `NoNewPrivileges`). Install:

```sh
cp deploy/haruspex.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now haruspex
```

Edit the `Environment=` lines for your adapter paths first. Without this (or an
equivalent — a container restart policy, a process manager), a crash is permanent until
someone notices.

## Observability

- `GET /health` — `{"ok": bool, "domains": [...], "default": "..."}`. `ok` is only
  `true` once the default domain's adapter has loaded — use it as a real readiness
  probe, not just a liveness check, if your orchestrator supports the distinction (a
  cold start can take tens of seconds to minutes depending on device and adapter count).
- `GET /metrics` — plain-text counters (`haruspex_requests_total`,
  `haruspex_requests_domain_<name>`, `haruspex_requests_rejected_oversized`,
  `haruspex_requests_errors`, `haruspex_requests_unauthorized`). No new dependency; a
  Prometheus scrape config can point at this path directly.
- `HARUSPEX_AUDIT_LOG=<path>` — one JSONL line per request: timestamp, resolved domain,
  routing source, and each question's label/probability/source. **The raw `state` is
  never logged** — it may be sensitive finding or source-code content; only its length
  is recorded. Rotate this file externally (logrotate or equivalent); Haruspex only appends.

## Throughput and device

Measured on this project's dev box (CPU-only, 5 domains loaded, `concurrency=12`
contending with another script on the same host): p50 4.6s, p95 37.7s, ~1.25 req/s.
**CPU serving is not production-volume-ready.** On GPU with no contention, single-request
latency was ~0.3–0.6s warm. If you expect more than occasional/interactive traffic:

- Serve on a CUDA device (`pick_device()` already prefers it automatically — just don't
  set `CUDA_VISIBLE_DEVICES=""`).
- Each adapter costs real memory (~2 GB CPU, less on GPU with fp16) — size the host/GPU
  for the number of domains you actually serve, not all five if you only need two.
- This project has not load-tested GPU serving under sustained concurrent traffic from
  multiple independent clients; treat the numbers above as the known floor, not a ceiling.

## Adapter versioning and rollback

There is no built-in registry today. Until there is one:

- Keep each trained adapter in its own directory/HF revision, named with a date or
  version (`runs/web-2026-10-17`, not `runs/web` overwritten in place).
- Before replacing a running adapter, keep the previous one on disk. Point
  `HARUSPEX_ADAPTERS` at the new path only after `haruspex.eval` / the sanity gate
  confirms it; rollback is "point the env var back and restart."
- `haruspex.json` records the base model + adapter subpath it was built from — enough
  to reproduce how a given directory was assembled, not a full lineage/registry.
