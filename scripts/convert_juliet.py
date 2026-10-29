"""Convert NIST Juliet Test Suite for Java 1.3 -> Haruspex boolean rows (train / cal / dev).

LICENSE: Juliet is a US-government work, public domain in the US (17 USC 105) and CC0
elsewhere (https://samate.nist.gov/SARD/test-suites/111), so derived rows may be trained
on and redistributed.

Juliet covers the OWASP Benchmark categories (SQLi, command injection, XSS, path traversal,
weak crypto / hash / random, LDAP, XPath, insecure cookie), so its rows ask exactly the
questions convert_owasp_benchmark.py asks. The Benchmark stays a test-only outside check.

Each single-file test case has one bad() method and several good*() ones. A row's state is
one of those entry methods plus the same-file helpers it calls (badSink, goodG2BSource ...)
and the class fields. Every comment is stripped (Juliet marks "FLAW" / "FIX" in comments)
and the bad / good / G2B / B2G name prefixes are rewritten, so the label is not readable
from names. Per test case: the bad method plus one good method, chosen with a fixed seed,
to keep the classes balanced. Splits are by functional variant (the file name without its
_NN flow suffix), so the same code shape never sits in two splits.

The Juliet code is only read as text; it is never compiled or run.

  python -I scripts/convert_juliet.py --src ~/datasets/juliet-java/src/Java/src/testcases
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Juliet CWE folder -> (OWASP Benchmark category, CWE the Benchmark files it under)
FOLDERS = {
    "CWE78_OS_Command_Injection": ("cmdi", 78),
    "CWE89_SQL_Injection": ("sqli", 89),
    "CWE80_XSS": ("xss", 79), "CWE81_XSS_Error_Message": ("xss", 79),
    "CWE83_XSS_Attribute": ("xss", 79),
    "CWE23_Relative_Path_Traversal": ("pathtraver", 22),
    "CWE36_Absolute_Path_Traversal": ("pathtraver", 22),
    "CWE327_Use_Broken_Crypto": ("crypto", 327),
    "CWE328_Reversible_One_Way_Hash": ("hash", 328),
    "CWE338_Weak_PRNG": ("weakrand", 330),
    "CWE614_Sensitive_Cookie_Without_Secure": ("securecookie", 614),
    "CWE90_LDAP_Injection": ("ldapi", 90),
    "CWE643_Xpath_Injection": ("xpathi", 643),
}
# Same wording as convert_owasp_benchmark.py, so the adapter learns the Benchmark's question.
CATEGORIES = {
    "cmdi": "OS command injection", "crypto": "weak encryption", "hash": "weak hashing",
    "ldapi": "LDAP injection", "pathtraver": "path traversal",
    "securecookie": "insecure cookie (missing Secure flag)", "sqli": "SQL injection",
    "weakrand": "insecure randomness", "xpathi": "XPath injection", "xss": "cross-site scripting",
}
FLOW = {"true": "attacker-controlled input reaches the sink without adequate validation "
                "or encoding",
        "false": "the input is safe, sanitized, or never reaches the sink"}
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
_METHOD = re.compile(r'^[ \t]*(?:public|private|protected)[^\n;=]*?\b(\w+)\s*\([^)]*\)[^{;]*\{', re.M)
_ENTRY = re.compile(r'^(bad|good(?:G2B|B2G)?\d*)$')
# bad / goodG2B1 / goodB2G / good1 as an identifier or an identifier prefix (badSink, goodG2BPrivate)
_PREFIX = re.compile(r'\b(?:bad|good(?:G2B|B2G)?\d*)(?=[A-Z_]\w*\b|\b)')
_CONSTS = ["alpha", "report", "default", "user01", "hello", "item", "config", "sample",
           "42", "guest", "admin", "data", "test", "value", "main", "x"]
_LEAK = re.compile(r'\b(?:bad|good|flaw|fix|g2b|b2g)\w*', re.I)


def strip_comments(src: str) -> str:
    return _TOKENS.sub(lambda m: m.group() if m.group()[0] in "\"'" else "", src)


def _block_end(src: str, i: int) -> int:
    """Index just past the brace block opening at src[i] == '{' (literals are skipped)."""
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c in "\"'":
            m = _TOKENS.match(src, j)
            j = m.end() if m else j + 1
            continue
        depth += c == "{"
        depth -= c == "}"
        j += 1
        if depth == 0:
            return j
    raise ValueError("unbalanced braces")


def split_class(src: str) -> tuple[dict[str, str], str]:
    """(methods by name, class-level field text) of the one top-level class."""
    body_start = src.index("{", src.index(" class ")) + 1
    body = src[body_start:_block_end(src, body_start - 1) - 1]
    methods, fields, pos = {}, [], 0
    for m in _METHOD.finditer(body):
        if m.start() < pos:
            continue                      # inside a method already taken
        end = _block_end(body, m.end() - 1)
        fields.append(body[pos:m.start()])
        methods[m.group(1)] = body[m.start():end]
        pos = end
    fields.append(body[pos:])
    text = "\n".join(ln for ln in "".join(fields).splitlines() if ln.strip())
    return methods, text


def state_for(entry: str, methods: dict[str, str], fields: str) -> str:
    """The entry method, the same-file methods it reaches, and the fields; names neutral."""
    seen, todo = [], [entry]
    while todo:
        name = todo.pop(0)
        if name in seen or name not in methods:
            continue
        seen.append(name)
        todo += [n for n in methods if n not in seen and re.search(rf"\b{n}\s*\(", methods[name])]
    parts = ([fields] if fields.strip() else []) + [methods[n] for n in seen]
    text = "\n\n".join(_dedent(p) for p in parts)

    def neutral(m):   # a bare name (also when quoted in a log string) -> handle; a prefix -> process
        nxt = m.string[m.end():m.end() + 1]
        return "process" if nxt.isupper() or nxt == "_" else "handle"
    return _PREFIX.sub(neutral, text)


def _dedent(block: str) -> str:
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    pad = min((len(ln) - len(ln.lstrip()) for ln in lines), default=0)
    return "\n".join(ln[pad:] for ln in lines)


def variant_of(stem: str) -> str:
    return re.sub(r"_\d+[a-z]?$", "", stem)


def split_of(variant: str) -> str:
    h = zlib.crc32(variant.encode()) % 10
    return "cal" if h == 0 else "dev" if h == 1 else "train"


def convert_file(path: Path, cat: str, cwe: int, rng: random.Random) -> list[dict]:
    methods, fields = split_class(strip_comments(path.read_text(encoding="utf-8", errors="replace")))
    entries = [n for n in methods if _ENTRY.match(n) and n != "good"]
    goods = sorted(n for n in entries if n != "bad")
    if "bad" not in entries or not goods:
        return []
    picks = [("bad", True), (rng.choice(goods), False)]
    # Juliet's hardcoded good source is always "foo": keep the constant, not the word.
    consts = [f'"{rng.choice(_CONSTS)}"' for _ in picks]
    return [{"id": f"juliet-{path.stem}-{name}", "side": "defensive", "category": f"owasp_{cat}",
             "source": "juliet_java_1.3", "type": "boolean",
             "state": state_for(name, methods, fields).replace('"foo"', c)
                      .replace('"Benign, fixed string"', '"done"'),
             "question": f"Is this code vulnerable to {CATEGORIES[cat]} (CWE-{cwe})?",
             "criteria": CRITERIA.get(cat, FLOW), "target": real} for (name, real), c in zip(picks, consts)]


def main():
    ap = argparse.ArgumentParser(description="Juliet Java 1.3 -> Haruspex boolean rows")
    ap.add_argument("--src", default="~/datasets/juliet-java/src/Java/src/testcases")
    ap.add_argument("--out-prefix", default=str(REPO / "data" / "juliet"))
    ap.add_argument("--max-chars", type=int, default=6000, help="drop states longer than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, rng = Path(args.src).expanduser(), random.Random(args.seed)
    splits: dict[str, list] = {"train": [], "cal": [], "dev": []}
    skipped = Counter()
    for folder, (cat, cwe) in FOLDERS.items():
        # single-file cases only: multi-file ones (_51a.java, _51b.java, ...) end in a letter
        for path in sorted((src / folder).rglob("*.java")):
            if not re.search(r"_\d+\.java$", path.name):
                skipped["multi-file"] += 1
                continue
            try:
                rows = convert_file(path, cat, cwe, rng)
            except ValueError:
                skipped["unparsed"] += 1
                continue
            for r in rows:
                if len(r["state"]) > args.max_chars:
                    skipped["too long"] += 1
                    continue
                splits[split_of(variant_of(path.stem))].append(r)
    leaks = sum(bool(_LEAK.search(r["state"])) for rs in splits.values() for r in rs)
    for name, rows in splits.items():
        rng.shuffle(rows)
        out = Path(f"{args.out_prefix}_{name}.jsonl")
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        cats = Counter(r["category"] for r in rows)
        print(f"[juliet] {name}: {len(rows)} rows ({sum(r['target'] for r in rows)} vulnerable) -> {out}")
        print("         " + ", ".join(f"{c[6:]}={n}" for c, n in sorted(cats.items())))
    print(f"[juliet] skipped: {dict(skipped)}; states still containing a label-like word: {leaks}")


if __name__ == "__main__":
    main()
