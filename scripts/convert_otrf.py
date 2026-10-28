"""Convert OTRF Security-Datasets (Windows atomic captures) -> Haruspex tactic choice rows.

OTRF Security-Datasets (github.com/OTRF/Security-Datasets, MIT) records Windows Security /
Sysmon / PowerShell logs while one documented attack runs in a lab. Each dataset has a
YAML in datasets/atomic/_metadata listing its ATT&CK mappings and an `adversary_view`:
the operator's console transcript, i.e. the commands that were actually typed.

Most events in a capture are background noise, so the capture's tactic is NOT pasted
onto every event. Labelling scheme:

  1. Label (per dataset): the tactic folder the authors filed the capture under
     (datasets/atomic/windows/<tactic>/host/...) and only when that tactic is also in
     the YAML attack_mappings. Two independent author annotations must agree; datasets
     where they disagree (or the folder is "other") are dropped.
  2. Attribution (per event): seed processes are Sysmon process creations (EID 1) whose
     command line is the documented command — image name appears in the adversary_view
     and >= SEED_OVERLAP of its argument tokens do too. The attacker set is the seeds
     plus all their descendants (ParentProcessGuid closure). Only events emitted BY those
     processes are kept: Sysmon events by ProcessGuid / SourceProcessGuid, Security and
     PowerShell events by (host, pid) when that pid is unambiguous and the event follows
     the process start. Ancestors (the C2 implant, the operator's shell) are excluded —
     their activity also includes beaconing / unrelated work. An attacker event whose
     (event id, target) a background process in the same capture also produced is
     dropped as commonplace (ntdll.dll loads, shared registry reads).
     The label is the tactic of the documented PROCEDURE, not a per-step judgment: a
     procedure's trigger steps (e.g. a reboot that fires a persistence task) carry it too.
  3. Datasets with no seed (in-process implant modules, no Sysmon EID 1) yield nothing.
  4. No benign rows: nothing in a lab capture is verified benign — background events are
     unlabelled, and include attack side-effects on other hosts.

A row is a window of up to WINDOW attacker events (time order, duplicate renders
removed) from one capture, rendered via evidence.from_windows_event + render(). Host and
non-builtin account names are redacted: they identify the lab, which correlates with the
author and therefore the tactic, and would be a shortcut that does not transfer.

Splits are by YAML dataset id (one id can cover several capture variants, which are
near-duplicates), content-hashed: test / dev / cal / train. Output is git-ignored.

  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/OTRF/Security-Datasets.git ~/datasets/otrf/repo
  git -C ~/datasets/otrf/repo sparse-checkout set --no-cone /LICENSE \
      '/datasets/atomic/_metadata/*' '/datasets/atomic/windows/*/host/*'
  python -I scripts/convert_otrf.py --repo ~/datasets/otrf/repo --out-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.evidence import from_windows_event, render  # noqa: E402

# Folder name -> ATT&CK tactic id. Descriptions are MITRE ATT&CK's own tactic summaries.
TACTICS = {
    "execution": ("TA0002", "the adversary is trying to run malicious code"),
    "persistence": ("TA0003", "the adversary is trying to maintain their foothold"),
    "privilege_escalation": ("TA0004", "the adversary is trying to gain higher-level permissions"),
    "defense_evasion": ("TA0005", "the adversary is trying to avoid being detected"),
    "credential_access": ("TA0006", "the adversary is trying to steal account names and passwords"),
    "discovery": ("TA0007", "the adversary is trying to figure out your environment"),
    "lateral_movement": ("TA0008", "the adversary is trying to move through your environment"),
    "collection": ("TA0009", "the adversary is trying to gather data of interest to their goal"),
}
CANDIDATES = {k: v[1] for k, v in TACTICS.items()}
QUESTION = "Which ATT&CK tactic does this attacker activity serve?"
SEED_OVERLAP = 0.6   # share of a command line's argument tokens found in the adversary_view
# (measured on all 100 Windows datasets: documented commands score >= 0.62, background
# processes whose image name happens to occur in the view score <= 0.33)
WINDOW = 8           # events per row
MAX_WINDOWS = 10     # rows per capture, so one noisy capture cannot dominate a tactic
_TOKEN = re.compile(r"[a-z0-9_][a-z0-9_.\-]*")
_TECHNIQUE_ID = re.compile(r"\bT1\d{3}(?:\.\d{3})?\b", re.I)
_BUILTIN = ("nt authority", "nt service", "font driver host", "window manager")


def tokens(text: str) -> set[str]:
    return {t.rstrip(".-") for t in _TOKEN.findall((text or "").lower())} - {""}


def label(meta: dict) -> str | None:
    """Tactic folder of the dataset's host files, if the YAML attack_mappings confirm it."""
    folders = {m.group(1) for f in meta.get("files") or []
               if (m := re.search(r"/atomic/windows/([a-z_]+)/host/", f.get("link", "")))}
    if len(folders) != 1:
        return None
    folder = folders.pop()
    mapped = {t for m in meta.get("attack_mappings") or [] for t in m.get("tactics") or []}
    return folder if folder in TACTICS and TACTICS[folder][0] in mapped else None


def _args(cmdline: str) -> str:
    """Command line without its leading image path."""
    c = cmdline.strip()
    if c.startswith('"'):
        end = c.find('"', 1)
        return c[end + 1:] if end > 0 else ""
    return c.split(None, 1)[1] if " " in c else ""


def is_seed(image: str, cmdline: str, view: set[str]) -> bool:
    """True when this process creation is a command documented in the adversary_view."""
    stem = re.split(r"[\\/]", (image or "").lower())[-1]
    if not stem or not (stem in view or stem.removesuffix(".exe") in view):
        return False
    arg = tokens(_args(cmdline or ""))
    # NOTE: an argument-less command (e.g. bare "whoami") is never a seed — its image name
    # alone is too weak to tell it from background. Its parent/child can still be.
    return bool(arg) and len(arg & view) / len(arg) >= SEED_OVERLAP


def _host(h) -> str:
    return str(h or "").split(".")[0].lower()


def _pid(v) -> int | None:
    try:
        s = str(v).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    except (TypeError, ValueError):
        return None


def flat(e: dict) -> dict:
    """Winlogbeat 6 shape (event_data / event_id / log_name) -> the flat NXLog shape."""
    if "event_data" in e and "EventID" not in e:
        return {**(e.get("event_data") or {}), "EventID": e.get("event_id"),
                "Channel": e.get("log_name"), "Hostname": e.get("computer_name"),
                "@timestamp": e.get("@timestamp"), "RecordNumber": e.get("record_number")}
    return e


def _sysmon(e: dict) -> bool:
    return "sysmon" in str(e.get("Channel", "")).lower()


def attacker_events(events: list[dict], view: set[str]) -> list[dict]:
    """Events emitted by documented-command processes and their descendants, time order."""
    events = sorted((flat(e) for e in events),
                    key=lambda e: (str(e.get("@timestamp") or ""), _pid(e.get("RecordNumber")) or 0))
    procs = [e for e in events if _sysmon(e) and str(e.get("EventID")) == "1" and e.get("ProcessGuid")]
    tree = {p["ProcessGuid"] for p in procs if is_seed(p.get("Image"), p.get("CommandLine"), view)}
    for p in procs:                       # time order: a parent is created before its child
        if p.get("ParentProcessGuid") in tree:
            tree.add(p["ProcessGuid"])
    # (host, pid) -> start time, for non-Sysmon channels; a pid started twice is ambiguous.
    starts, seen = {}, Counter((_host(p.get("Hostname")), _pid(p.get("ProcessId"))) for p in procs)
    for p in procs:
        k = (_host(p.get("Hostname")), _pid(p.get("ProcessId")))
        if p["ProcessGuid"] in tree and seen[k] == 1:
            starts[k] = str(p.get("@timestamp") or "")
    out, background = [], set()
    for e in events:
        if _sysmon(e):
            g = e.get("SourceProcessGUID") or e.get("SourceProcessGuid") or e.get("ProcessGuid")
            mine = g in tree
        else:
            pid = _pid(e.get("NewProcessId") if str(e.get("EventID")) == "4688" else
                       e.get("ExecutionProcessID") or e.get("ProcessId") or e.get("ProcessID"))
            start = starts.get((_host(e.get("Hostname")), pid))
            mine = start is not None and str(e.get("@timestamp") or "") >= start
        if mine:
            out.append(e)
        else:
            background.add(_target(e))
    # An attacker event whose (event id, target) any background process in the same capture
    # also produced (ntdll.dll loads, common registry reads) says nothing about the attack.
    return [e for e in out if _target(e)[1] == "" or _target(e) not in background]


def _target(e: dict) -> tuple[str, str]:
    return str(e.get("EventID")), str(from_windows_event(e).get("title", "")).lower()


def canonical(e: dict) -> dict:
    """Canonical evidence with lab identifiers (host, non-builtin accounts) redacted."""
    ev = from_windows_event(e)
    ev.pop("host", None)
    if ev.get("user") and not ev["user"].lower().startswith(_BUILTIN):
        ev["user"] = "DOMAIN\\user" if "\\" in ev["user"] else "user"
    return ev


def windows(events: list[dict]) -> list[str]:
    seen, evs = set(), []
    for e in events:
        r = render(canonical(e))
        if r not in seen:
            seen.add(r)
            evs.append(canonical(e))
    chunks = [evs[i:i + WINDOW] for i in range(0, len(evs), WINDOW)]
    # NOTE: keeps the first MAX_WINDOWS (the attack's start); an even stride would sample
    # later phases too but rarely matters at these capture sizes.
    # Atomic Red Team / PurpleSharp put ATT&CK ids in file names and arguments: a label
    # leak that never occurs in the wild, so the ids are masked.
    return [_TECHNIQUE_ID.sub("T####", render(c)) for c in chunks[:MAX_WINDOWS]]


def read_archive(path: Path) -> tuple[list[dict], bool]:
    """JSON-lines events from a .zip / .tar.gz capture; (events, complete?)."""
    lines, complete = [], True
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    if n.endswith(".json"):
                        lines += z.read(n).decode("utf-8", "replace").splitlines()
        else:
            with tarfile.open(path, "r:gz") as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".json"):
                        lines += t.extractfile(m).read().decode("utf-8", "replace").splitlines()
    except (EOFError, OSError, tarfile.TarError, zipfile.BadZipFile):
        complete = False                  # upstream archive is truncated: keep whole lines
    out = []
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out, complete


def split_of(dataset_id: str, seed: int) -> str:
    u = int(hashlib.sha256(f"{seed}:{dataset_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    return "test" if u < 0.2 else "dev" if u < 0.3 else "cal" if u < 0.4 else "train"


def main():
    ap = argparse.ArgumentParser(description="OTRF Security-Datasets -> Haruspex tactic rows")
    ap.add_argument("--repo", required=True, help="sparse checkout of OTRF/Security-Datasets")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    repo, out = Path(args.repo).expanduser(), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = defaultdict(list), Counter()
    for ypath in sorted((repo / "datasets/atomic/_metadata").glob("SDWIN-*.yaml")):
        meta = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        tactic = label(meta)
        if tactic is None:
            skipped["label disagreement"] += 1
            continue
        view = tokens((meta.get("simulation") or {}).get("adversary_view") or "")
        split, n = split_of(meta["id"], args.seed), 0
        for f in meta.get("files") or []:
            rel = f.get("link", "").split("/master/", 1)[-1]
            path = repo / rel
            if "/host/" not in rel or not path.is_file():
                continue
            events, complete = read_archive(path)
            if not complete:
                print(f"[otrf] {meta['id']}: {path.name} truncated upstream, using whole lines")
            for i, state in enumerate(windows(attacker_events(events, view))):
                rows[split].append({
                    "id": f"otrf-{split}-{meta['id']}-{path.stem}-{i}", "side": "defensive",
                    "category": "def_attack_tactic", "source": "otrf", "type": "choice",
                    "state": state, "question": QUESTION, "candidates": CANDIDATES,
                    "target": tactic})
                n += 1
        if not n:
            skipped["no documented-command process"] += 1
    print(f"[otrf] datasets skipped: {dict(skipped)}")
    for split in ("train", "dev", "cal", "test"):
        path = out / f"otrf_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows[split]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        caps = len({r["id"].split("-")[2] + r["id"].split("-")[3] for r in rows[split]})
        print(f"[otrf] {split}: {len(rows[split])} rows from {caps} datasets "
              f"{dict(Counter(r['target'] for r in rows[split]))} -> {path}")


if __name__ == "__main__":
    main()
