"""One evidence schema for every log source, so the model never learns a vendor format.

Raw telemetry differs per product (Defender, Sentinel, Elastic ECS, Windows EVTX /
Sysmon, ...) but a triage decision rests on the same facts: which tactic/technique,
which entities in which role, what process/user/host/network, what automated verdicts.
`normalize()` maps a known raw shape onto those canonical fields and `render()` turns
them into the one compact text form the adapters are trained on. Supporting a new
source means adding a mapper here — not new training data in a new format.

Canonical evidence (all fields optional):
  source, kind, title, severity, outcome,
  tactics[], techniques[], verdicts[], threat_family[],
  event   {code, action, channel},
  process {image, command_line, parent_image},
  user, host,
  network {src_ip, dst_ip, dst_port, protocol, direction},
  entities[{type, role, value}],
  alerts, items          (counts, for aggregates such as an incident)

NOTE (ceiling): detection is by key shape, first match wins. A source whose JSON
matches none returns None and the state is passed through as JSON — the model then
sees an unfamiliar format, which is what the novelty gate (ood.py) is for.
"""

from __future__ import annotations

import json
from collections import Counter

_MAX_LIST = 12     # distinct values listed per field
_MAX_VALUE = 300   # chars per free-text value (command lines can be huge)
_MAX_EVENTS = 20   # events rendered from a list


def _s(v) -> str:
    v = "" if v is None else str(v).strip()
    return v[:_MAX_VALUE] + "..." if len(v) > _MAX_VALUE else v


def _list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.startswith("[") else v.replace(";", ",").split(",")
        except ValueError:
            v = [v]
    if not isinstance(v, (list, tuple)):
        v = [v]
    return [_s(x) for x in v if _s(x)]


def _get(d, path: str):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _prune(ev: dict) -> dict:
    out = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            v = {a: b for a, b in v.items() if b not in (None, "", [])}
        if v not in (None, "", [], {}):
            out[k] = v
    return out


# ---- mappers: raw shape -> canonical ---------------------------------------------

def from_ecs(d: dict) -> dict:
    """Elastic Common Schema document (Elastic Security, Winlogbeat, Filebeat)."""
    return _prune({
        "source": _s(_get(d, "observer.product") or _get(d, "agent.type") or "Elastic"),
        "kind": _s(_get(d, "event.kind") or "event"),
        "title": _s(_get(d, "rule.name") or _get(d, "kibana.alert.rule.name") or _get(d, "message")),
        "severity": _s(_get(d, "event.severity") or _get(d, "kibana.alert.severity")),
        "outcome": _s(_get(d, "event.outcome")),
        "tactics": _list(_get(d, "threat.tactic.name")),
        "techniques": _list(_get(d, "threat.technique.id")),
        "event": {"code": _s(_get(d, "event.code")), "action": _s(_get(d, "event.action")),
                  "channel": _s(_get(d, "winlog.channel") or _get(d, "event.dataset"))},
        "process": {"image": _s(_get(d, "process.executable") or _get(d, "process.name")),
                    "command_line": _s(_get(d, "process.command_line")),
                    "parent_image": _s(_get(d, "process.parent.executable") or _get(d, "process.parent.name"))},
        "user": _s(_get(d, "user.name")), "host": _s(_get(d, "host.name")),
        "network": {"src_ip": _s(_get(d, "source.ip")), "dst_ip": _s(_get(d, "destination.ip")),
                    "dst_port": _s(_get(d, "destination.port")),
                    "protocol": _s(_get(d, "network.transport") or _get(d, "network.protocol")),
                    "direction": _s(_get(d, "network.direction"))},
    })


def from_windows_event(d: dict) -> dict:
    """Windows Security / Sysmon event as JSON: flat (NXLog, OTRF Security-Datasets,
    Winlogbeat raw) or evtx_dump nested (Event.System + Event.EventData)."""
    if isinstance(d.get("Event"), dict):
        sysd = d["Event"].get("System") or {}
        data = d["Event"].get("EventData") or {}
        eid = sysd.get("EventID")
        eid = eid.get("#text") if isinstance(eid, dict) else eid
        d = {**data, "EventID": eid, "Channel": sysd.get("Channel"),
             "Hostname": sysd.get("Computer")}
    user = d.get("TargetUserName") or d.get("User") or d.get("SubjectUserName")
    dom = d.get("TargetDomainName") or d.get("SubjectDomainName")
    return _prune({
        "source": "Windows", "kind": "event",
        "event": {"code": _s(d.get("EventID")), "channel": _s(d.get("Channel")),
                  # NXLog ships Task as a bare numeric code ("7" for Sysmon 7): not an action.
                  "action": _s(next((v for v in (d.get("Task"), d.get("Category"), d.get("EventType"))
                                     if v not in (None, "") and not str(v).strip().isdigit()), ""))},
        "outcome": _s(d.get("Keywords") if "Audit" in str(d.get("Keywords", "")) else ""),
        "process": {"image": _s(d.get("Image") or d.get("NewProcessName") or d.get("ProcessName")),
                    "command_line": _s(d.get("CommandLine")),
                    "parent_image": _s(d.get("ParentImage") or d.get("ParentProcessName"))},
        "user": _s(f"{dom}\\{user}" if dom and user else user),
        "host": _s(d.get("Hostname") or d.get("Computer")),
        "network": {"src_ip": _s(d.get("IpAddress") or d.get("SourceIp")),
                    "dst_ip": _s(d.get("DestinationIp")),
                    "dst_port": _s(d.get("DestinationPort")),
                    "protocol": _s(d.get("Protocol"))},
        "entities": [{"type": "logon", "role": "", "value": f'type {d["LogonType"]}'}]
                    if d.get("LogonType") not in (None, "", "-") else [],
        "title": _s(d.get("TargetImage") or d.get("TargetObject") or d.get("ServiceName") or d.get("TaskName")
                    or d.get("ObjectName") or d.get("ImageLoaded") or d.get("TargetFilename")),
    })


def from_defender_alert(d: dict) -> dict:
    """Microsoft Graph security alert (alerts_v2: Defender XDR / MDE / MDO / Entra)."""
    ents = []
    for e in d.get("evidence") or []:
        t = str(e.get("@odata.type", "")).rsplit(".", 1)[-1].replace("Evidence", "")
        ents.append({"type": t, "role": ",".join(e.get("roles") or []) or "related",
                     "value": _s(e.get("fileName") or _get(e, "fileDetails.fileName")
                                 or e.get("ipAddress") or e.get("url") or e.get("deviceDnsName")
                                 or _get(e, "userAccount.userPrincipalName"))})
    return _prune({
        "source": _s(d.get("serviceSource") or "Microsoft Defender XDR"), "kind": "alert",
        "title": _s(d.get("title")), "severity": _s(d.get("severity")),
        "tactics": _list(d.get("category")), "techniques": _list(d.get("mitreTechniques")),
        "verdicts": _list([e.get("verdict") for e in d.get("evidence") or []
                           if e.get("verdict") not in (None, "unknown")]),
        "threat_family": _list(d.get("threatFamilyName")),
        "entities": ents,
    })


def from_sentinel_alert(d: dict) -> dict:
    """Microsoft Sentinel SecurityAlert row."""
    ents = []
    for e in _json_list(d.get("Entities")):
        if isinstance(e, dict) and e.get("Type"):
            ents.append({"type": _s(e["Type"]), "role": "",
                         "value": _s(e.get("Name") or e.get("Address") or e.get("Url")
                                     or e.get("HostName") or e.get("CommandLine"))})
    return _prune({
        "source": _s(d.get("ProductName") or "Microsoft Sentinel"), "kind": "alert",
        "title": _s(d.get("AlertName") or d.get("DisplayName")),
        "severity": _s(d.get("AlertSeverity")),
        "tactics": _list(d.get("Tactics")), "techniques": _list(d.get("Techniques")),
        "entities": ents,
    })


def _json_list(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return []
    return v if isinstance(v, list) else []


def _is_canonical(d: dict) -> bool:
    return d.get("schema") == "haruspex.evidence/1"


def normalize(obj) -> dict | list | None:
    """Raw source JSON -> canonical evidence (a list for a list of events), or None
    when the shape is not a known source."""
    if isinstance(obj, list):
        evs = [normalize(x) for x in obj[:_MAX_EVENTS] if isinstance(x, dict)]
        return evs if evs and all(isinstance(e, dict) for e in evs) else None
    if not isinstance(obj, dict):
        return None
    if _is_canonical(obj):
        return obj
    if "evidence" in obj and ("mitreTechniques" in obj or "serviceSource" in obj):
        return from_defender_alert(obj)
    if "AlertName" in obj and ("Tactics" in obj or "Entities" in obj):
        return from_sentinel_alert(obj)
    if "EventID" in obj or isinstance(_get(obj, "Event.System"), dict):
        return from_windows_event(obj)
    if isinstance(obj.get("event"), dict) or "ecs" in obj:
        return from_ecs(obj)
    return None


# ---- canonical -> text ------------------------------------------------------------

def _fmt(values) -> str:
    c = Counter(v for v in values if v)
    return ", ".join(f"{k} x{n}" if n > 1 else k for k, n in c.most_common(_MAX_LIST))


def render(ev) -> str:
    """Canonical evidence (or a list of it) -> the compact text the model reads."""
    if isinstance(ev, list):
        return "\n".join(f"[{i + 1}] " + render(e).replace("\n", " ") for i, e in enumerate(ev))
    head = " ".join(x for x in (ev.get("source"), ev.get("kind") or "evidence") if x)
    counts = []
    if ev.get("alerts"):
        counts.append(f'{ev["alerts"]} alert(s)')
    if ev.get("items"):
        counts.append(f'{ev["items"]} evidence item(s)')
    lines = [head + (" with " + " and ".join(counts) if counts else "")
             + (f': {ev["title"]}' if ev.get("title") else "") + "."]
    e = ev.get("event") or {}
    if e:
        lines.append("Event: " + " ".join(x for x in (
            e.get("code"), e.get("action"), f'({e["channel"]})' if e.get("channel") else "") if x) + ".")
    for key, label in (("severity", "Severity"), ("outcome", "Outcome")):
        if ev.get(key):
            lines.append(f"{label}: {ev[key]}.")
    p = ev.get("process") or {}
    if p:
        lines.append("Process: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in p.items()) + ".")
    for key, label in (("user", "User"), ("host", "Host")):
        if ev.get(key):
            lines.append(f"{label}: {ev[key]}.")
    n = ev.get("network") or {}
    if n:
        lines.append("Network: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in n.items()) + ".")
    ents = ev.get("entities") or []
    if ents:
        lines.append("Evidence: " + _fmt(
            " ".join(x for x in (x.get("type"), f'({x["role"].lower()})' if x.get("role") else "",
                                 x.get("value") or "") if x) for x in ents) + ".")
    for key, label in (("tactics", "Alert categories"), ("techniques", "MITRE techniques"),
                       ("verdicts", "Automated verdicts"), ("threat_family", "Threat family")):
        if ev.get(key):
            lines.append(f"{label}: {_fmt(ev[key])}.")
    for key, val in (ev.get("extra") or {}).items():
        lines.append(f"{key}: {_fmt(_list(val))}.")
    return "\n".join(lines)


def to_text(state) -> str | None:
    """Rendered text when `state` is a known source shape, else None."""
    ev = normalize(state)
    return render(ev) if ev else None
