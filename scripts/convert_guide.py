"""Convert Microsoft GUIDE (Defender XDR incident triage) -> Haruspex choice rows.

GUIDE (Freitas et al. 2024, "AI-Driven Guided Response for Security Operation Centers
with Microsoft Copilot for Security") ships one CSV row per piece of alert evidence,
graded per incident by SOC analysts: TruePositive / BenignPositive / FalsePositive.

One row per incident (OrgId, IncidentId): its evidence is folded into a plain-text
summary of the fields a real Defender/Sentinel export also carries — alert categories,
MITRE techniques, entity types + evidence roles, automated verdicts, threat family, mail
direction, cloud resource types. Deliberately dropped:
  - anonymised integer ids (DetectorId, AlertTitle, DeviceId, ...): memorising them
    does not transfer to anyone else's tenant;
  - ActionGrouped / ActionGranular: remediation happens AFTER triage, so it leaks the
    grade;
  - "DomainPII_<hash>" placeholder values.

Splits: GUIDE_Train.csv -> train / dev / cal by a content hash of the incident key;
GUIDE_Test.csv -> test (Microsoft's own held-out file). Each split is class-balanced by
capping rows per grade. Output is git-ignored (data/*.jsonl); GUIDE is CDLA-Permissive-2.0.

  python scripts/convert_guide.py --src ~/datasets/guide --out-dir data --per-grade 600
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.evidence import render  # noqa: E402

GRADES = {"TruePositive": "true_positive", "BenignPositive": "benign_positive",
          "FalsePositive": "false_positive"}
CANDIDATES = {
    "true_positive": "real malicious activity that needs response",
    "benign_positive": "the detection is correct but the activity is expected or authorised "
                       "(admin work, testing, sanctioned tools)",
    "false_positive": "the detection fired on activity that does not match what it looks for",
}
QUESTION = "What is the triage grade for this security incident?"
# Multi-valued fields folded per incident: (csv column, label in the summary).
_FIELDS = (("Category", "Alert categories"), ("MitreTechniques", "MITRE techniques"),
           ("LastVerdict", "Automated verdicts"), ("SuspicionLevel", "Suspicion level"),
           ("ThreatFamily", "Threat family"), ("AntispamDirection", "Mail direction"),
           ("ResourceType", "Cloud resources"), ("Roles", "Entity roles"))


def _clean(v: str) -> str:
    v = (v or "").strip()
    return "" if not v or v.startswith("DomainPII_") else v


def _bucket(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def fold(csv_path: Path, keep, max_incidents: int = 0) -> dict:
    """{incident_key: aggregate} over the evidence rows of incidents `keep(key)` accepts."""
    inc: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            grade = GRADES.get(r.get("IncidentGrade", ""))
            key = f'{r.get("OrgId")}:{r.get("IncidentId")}'
            if grade is None or not keep(key):
                continue
            a = inc.get(key)
            if a is None:
                if max_incidents and len(inc) >= max_incidents:
                    continue
                a = inc[key] = {"grade": Counter(), "alerts": set(), "evidence": Counter(),
                                **{c: Counter() for c, _ in _FIELDS}}
            a["grade"][grade] += 1
            a["alerts"].add(r.get("AlertId"))
            ent = _clean(r.get("EntityType", ""))
            if ent:
                a["evidence"][(ent, _clean(r.get("EvidenceRole", "")).lower())] += 1
            for col, _ in _FIELDS:
                for v in _clean(r.get(col, "")).split(";"):
                    if v.strip():
                        a[col][v.strip()] += 1
    return inc


def canonical(a: dict) -> dict:
    """Folded incident -> canonical evidence (haruspex/evidence.py)."""
    el = lambda col: list(a[col].elements())
    extra = {label: el(col) for col, label in _FIELDS[3:] if col not in ("ThreatFamily",) and a[col]}
    return {"schema": "haruspex.evidence/1", "source": "Microsoft Defender XDR", "kind": "incident",
            "alerts": len(a["alerts"]), "items": sum(a["evidence"].values()),
            "entities": [{"type": t, "role": r} for (t, r) in a["evidence"].elements()],
            "tactics": el("Category"), "techniques": el("MitreTechniques"),
            "verdicts": el("LastVerdict"), "threat_family": el("ThreatFamily"), "extra": extra}


def to_row(key: str, a: dict, split: str) -> dict:
    return {"id": f"guide-{split}-{key}", "side": "defensive", "category": "def_incident_grade",
            "source": "guide", "type": "choice", "state": render(canonical(a)), "question": QUESTION,
            "candidates": CANDIDATES, "target": a["grade"].most_common(1)[0][0]}


def balanced(inc: dict, per_grade: int, seed: int, split: str) -> list[dict]:
    by: dict[str, list] = {g: [] for g in CANDIDATES}
    for key in sorted(inc):
        a = inc[key]
        by[a["grade"].most_common(1)[0][0]].append(key)
    # Identical summaries are KEPT: their repeats carry how often each grade occurs for
    # that pattern; deduping would teach 50/50 on a pattern that is 90/10.
    rng, out = random.Random(seed), []
    for keys in by.values():
        rng.shuffle(keys)
        out += [to_row(k, inc[k], split) for k in keys[:per_grade]]
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Microsoft GUIDE -> Haruspex incident-grade rows")
    ap.add_argument("--src", required=True, help="dir holding GUIDE_Train.csv / GUIDE_Test.csv")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--per-grade", type=int, default=600, help="train rows per grade")
    ap.add_argument("--eval-per-grade", type=int, default=200, help="dev / cal rows per grade")
    ap.add_argument("--test-per-grade", type=int, default=500)
    ap.add_argument("--sample", type=float, default=0.05,
                    help="fraction of incidents to fold (bounds memory on the 2.4 GB file)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src).expanduser(), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # One pass over the train file; the hash sends each incident to exactly one split.
    inc = fold(src / "GUIDE_Train.csv", lambda k: _bucket(k, args.seed) < args.sample)
    parts = {"train": {}, "dev": {}, "cal": {}}
    for k, a in inc.items():
        u = _bucket(k, args.seed + 1)
        parts["dev" if u < 0.15 else "cal" if u < 0.30 else "train"][k] = a
    test = fold(src / "GUIDE_Test.csv", lambda k: _bucket(k, args.seed) < args.sample)

    caps = {"train": args.per_grade, "dev": args.eval_per_grade, "cal": args.eval_per_grade}
    for split, rows in [(s, balanced(parts[s], caps[s], args.seed, s)) for s in parts] + \
                       [("test", balanced(test, args.test_per_grade, args.seed, "test"))]:
        path = out / f"guide_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[guide] {split}: {len(rows)} rows "
              f"{dict(Counter(r['target'] for r in rows))} -> {path}")


if __name__ == "__main__":
    main()
