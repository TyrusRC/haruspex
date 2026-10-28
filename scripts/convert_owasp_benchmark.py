"""Convert the OWASP Benchmark v1.2 (Java) -> Haruspex boolean rows. EXTERNAL TEST SET ONLY.

LICENSE: the OWASP Benchmark is GPL-2.0. Rows derived from it embed its source code, so
they must NOT be committed to this repo or redistributed with the model or its datasets.
This script therefore writes only a test file, and only OUTSIDE the repo (default
~/datasets/owasp-benchmark-rows/owasp_test.jsonl); it refuses an --out inside the repo.
It never writes train / dev / cal rows: the Benchmark is for measuring, not for training.

Each test case (expectedresults-1.2.csv: name, category, real vulnerability, CWE) becomes
one boolean row. The state is BenchmarkTestNNNNN.java from `public void doPost` to the end
of the class (the license header, imports and the boilerplate doGet are dropped) with ALL
comments removed: the sources carry hints such as "condition 'B', which is safe" or
"this whole flow is 'safe'". When the code reads benchmark.properties (hashAlg1,
cryptoAlg2, ...), the values it reads are appended, because the in-code defaults differ
from the real values and decide the answer.

The Benchmark code is only read as text; it is never compiled or run.

  python -I scripts/convert_owasp_benchmark.py --src ~/datasets/owasp-benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "cmdi": "OS command injection", "crypto": "weak encryption", "hash": "weak hashing",
    "ldapi": "LDAP injection", "pathtraver": "path traversal",
    "securecookie": "insecure cookie (missing Secure flag)", "sqli": "SQL injection",
    "trustbound": "trust boundary violation", "weakrand": "insecure randomness",
    "xpathi": "XPath injection", "xss": "cross-site scripting",
}
FLOW = {"true": "attacker-controlled input reaches the sink without adequate validation "
                "or encoding",
        "false": "the input is safe, sanitized, or never reaches the sink"}
# Categories that are about an API choice, not a data flow.
CRITERIA = {
    "crypto": {"true": "the data is encrypted with a weak cipher or mode",
               "false": "a strong cipher and mode are used"},
    "hash": {"true": "a weak hash algorithm is used", "false": "a strong hash algorithm is used"},
    "weakrand": {"true": "a predictable random generator is used where security needs "
                         "unpredictability",
                 "false": "a cryptographically secure random generator is used"},
    "securecookie": {"true": "a cookie is set without the Secure flag",
                     "false": "the cookie is set with the Secure flag"},
}
_TOKENS = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
_PROP = re.compile(r'getProperty\(\s*"([^"]+)"')


def read_expected(path: Path) -> list[tuple[str, str, bool, int]]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for rec in csv.reader(line for line in fh if line.strip() and not line.startswith("#")):
            name, cat, real, cwe = (f.strip() for f in rec[:4])
            if cat not in CATEGORIES or real not in ("true", "false"):
                raise SystemExit(f"[owasp] unexpected row {rec!r}")   # fail closed
            out.append((name, cat, real == "true", int(cwe)))
    return out


def strip_comments(src: str) -> str:
    """Drop // and /* */ comments; string and char literals (e.g. "http://...") are kept."""
    return _TOKENS.sub(lambda m: m.group() if m.group()[0] in "\"'" else "", src)


def extract_state(java: str, props: dict[str, str]) -> str:
    i = java.find("public void doPost")
    if i < 0:
        raise ValueError("no doPost")
    body = strip_comments(java[java.rfind("\n", 0, i) + 1:]).rstrip()
    body = body[:-1] if body.endswith("}") else body   # the class's closing brace
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    state = textwrap.dedent("\n".join(lines)).strip()
    used = [k for k in dict.fromkeys(_PROP.findall(state)) if k in props]
    if used:
        state += "\n\nbenchmark.properties:\n" + "\n".join(f"{k}={props[k]}" for k in used)
    return state


def read_props(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (ln.split("=", 1) for ln in path.read_text(encoding="utf-8").splitlines()
             if "=" in ln and not ln.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def to_row(name: str, cat: str, real: bool, cwe: int, state: str) -> dict:
    return {"id": f"owasp-test-{name}", "side": "defensive", "category": f"owasp_{cat}",
            "source": "owasp_benchmark", "type": "boolean", "state": state,
            "question": f"Is this code vulnerable to {CATEGORIES[cat]} (CWE-{cwe})?",
            "criteria": CRITERIA.get(cat, FLOW), "target": real}


def main():
    ap = argparse.ArgumentParser(description="OWASP Benchmark -> Haruspex external test rows")
    ap.add_argument("--src", default="~/datasets/owasp-benchmark")
    ap.add_argument("--out", default="~/datasets/owasp-benchmark-rows/owasp_test.jsonl")
    ap.add_argument("--budget-chars", type=int, default=2000,
                    help="~512 model tokens; only used to report how many states overflow")
    args = ap.parse_args()
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser().resolve()
    if out.is_relative_to(REPO):
        raise SystemExit(f"[owasp] refusing to write GPL-derived rows inside the repo: {out}")
    code = src / "src/main/java/org/owasp/benchmark/testcode"
    props = read_props(src / "src/main/resources/benchmark.properties")

    rows = [to_row(n, c, r, w, extract_state((code / f"{n}.java").read_text(encoding="utf-8"),
                                             props))
            for n, c, r, w in read_expected(src / "expectedresults-1.2.csv")]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = sorted(len(r["state"]) for r in rows)
    q = statistics.quantiles(lens, n=10)
    over = sum(n > args.budget_chars for n in lens)
    print(f"[owasp] test: {len(rows)} rows -> {out}")
    print("[owasp] per category (true/false): " + ", ".join(
        f"{c}={sum(r['target'] for r in rows if r['category'] == f'owasp_{c}')}/"
        f"{sum(not r['target'] for r in rows if r['category'] == f'owasp_{c}')}" for c in CATEGORIES))
    print(f"[owasp] state chars: min {lens[0]}  p10 {q[0]:.0f}  median {statistics.median(lens):.0f}"
          f"  p90 {q[-1]:.0f}  max {lens[-1]}")
    print(f"[owasp] over {args.budget_chars} chars (truncated by the model): {over} "
          f"({100 * over / len(lens):.1f}%)")


if __name__ == "__main__":
    main()
