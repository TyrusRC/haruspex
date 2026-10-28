"""Fetch CVE descriptions + NIST-assigned CVSS v3 severity from the NVD API 2.0.

Writes one compact JSON record per CVE: {"id", "published", "description",
"severity"} — only CVEs whose CVSS v3.x score was assigned by NVD itself (source type
"Primary"), so every label is an independent NIST analyst judgment. Resumable: re-running
continues from the last saved page (a page cut off mid-write may repeat; convert_nvd.py
drops repeated ids).

Respects the published unauthenticated rate limit (5 requests / 30 s) by pausing 6 s
between pages. An API key (NVD_API_KEY env var, sent as the apiKey header) raises the
limit; the pause stays, conservatively.

  python scripts/fetch_nvd.py --out ~/datasets/nvd/cves.jsonl --since 2018-01-01
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE = 2000          # the API's maximum resultsPerPage
PAUSE_S = 6.0        # 5 requests per rolling 30 s window, unauthenticated
WINDOW_DAYS = 120    # the API's maximum publication-date range


def record(cve: dict) -> dict | None:
    """One NVD `cve` object -> {"id","published","description","severity"}, or None
    when there is no English description or no NVD-assigned (Primary) CVSS v3 severity."""
    desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
    if not desc or desc.startswith("** REJECT"):
        return None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        for m in metrics.get(key, []):
            sev = m.get("cvssData", {}).get("baseSeverity")
            if m.get("type") == "Primary" and sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                return {"id": cve["id"], "published": cve.get("published", ""),
                        "description": desc.strip(), "severity": sev.lower()}
    return None


def _get(url: str, key: str, tries: int = 6) -> dict:
    headers = {"User-Agent": "haruspex-dataset-builder"}
    if key:
        headers["apiKey"] = key
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
            wait = PAUSE_S * 2 ** attempt   # back off on 403/503 and network errors
            print(f"[nvd] {e} — retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"[nvd] giving up on {url}")


def windows(since: date, until: date, days: int = WINDOW_DAYS):
    """Contiguous [start, end] publication-date windows covering since..until; the API
    caps one pubStartDate/pubEndDate range at 120 days."""
    while since <= until:
        end = min(since + timedelta(days=days - 1), until)
        yield since, end
        since = end + timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description="NVD CVE descriptions + CVSS v3 severity")
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="2018-01-01",
                    help="first publication date (YYYY-MM-DD); older CVEs rarely carry an "
                         "NVD-assigned CVSS v3 score, so fetching them is mostly wasted")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    state = out.with_suffix(".state")   # "<window start>,<index>" resume point
    done_win, start = "", 0
    if state.exists():
        done_win, idx = state.read_text().split(",")
        start = int(idx)
    key = os.environ.get("NVD_API_KEY", "")
    with out.open("a", encoding="utf-8") as fh:
        for lo, hi in windows(date.fromisoformat(args.since), date.today()):
            if done_win and lo.isoformat() < done_win:
                continue
            if lo.isoformat() != done_win:
                start = 0
            total = None
            while total is None or start < total:
                rng = (f"pubStartDate={lo.isoformat()}T00:00:00.000&"
                       f"pubEndDate={hi.isoformat()}T23:59:59.999")
                page = _get(f"{API}?{rng}&startIndex={start}&resultsPerPage={PAGE}", key)
                total = page.get("totalResults", 0)
                kept = 0
                for v in page.get("vulnerabilities", []):
                    rec = record(v.get("cve", {}))
                    if rec:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept += 1
                fh.flush()
                start += len(page.get("vulnerabilities", [])) or PAGE
                state.write_text(f"{lo.isoformat()},{start}")   # after the page is saved
                print(f"[nvd] {lo}..{hi} {min(start, total)}/{total}  kept {kept}", flush=True)
                time.sleep(PAUSE_S)
            done_win = lo.isoformat()
    print(f"[nvd] done -> {out}")


if __name__ == "__main__":
    main()
