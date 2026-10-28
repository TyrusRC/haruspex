"""Evidence normalisation, novelty gate and long-state windowing (no model needed)."""

import tempfile
import unittest

from haruspex.data import decision_text
from haruspex.evidence import normalize, render, to_text
from haruspex.model import CHUNK_CHARS, MAX_WINDOWS, Haruspex
from haruspex.ood import NoveltyGate

BOOL = {"type": "boolean", "question": "Is this malicious?",
        "criteria": {"true": "t", "false": "f"}}


class EvidenceTest(unittest.TestCase):
    def test_two_sources_render_the_same_facts_the_same_way(self):
        # the whole point: Elastic and a Windows event for one process start look alike
        ecs = {"event": {"code": "4688"}, "process": {"command_line": "cmd.exe /c whoami"},
               "user": {"name": "alice"}, "host": {"name": "ws01"}}
        win = {"EventID": 4688, "CommandLine": "cmd.exe /c whoami",
               "TargetUserName": "alice", "Computer": "ws01"}
        a, b = to_text(ecs).splitlines()[1:], to_text(win).splitlines()[1:]
        self.assertEqual(a, b)

    def test_nested_evtx_dump(self):
        ev = normalize({"Event": {"System": {"EventID": {"#text": "1"}, "Computer": "h"},
                                  "EventData": {"Image": "C:\\x.exe"}}})
        self.assertEqual(ev["event"]["code"], "1")
        self.assertEqual(ev["process"]["image"], "C:\\x.exe")

    def test_unknown_shape_falls_back_to_json(self):
        self.assertIsNone(to_text({"foo": 1}))
        self.assertIn('"foo"', decision_text({"state": {"foo": 1}, "question": "q"}))

    def test_long_values_are_capped(self):
        txt = to_text({"EventID": 1, "CommandLine": "A" * 5000})
        self.assertLess(len(txt), 1000)

    def test_cloudtrail(self):
        ev = normalize({"eventSource": "iam.amazonaws.com", "eventName": "CreateAccessKey",
                        "userIdentity": {"type": "IAMUser", "arn": "arn:aws:iam::1:user/bob"},
                        "sourceIPAddress": "203.0.113.5", "errorCode": "AccessDenied",
                        "awsRegion": "us-east-1"})
        self.assertEqual(ev["source"], "AWS CloudTrail")
        self.assertEqual(ev["event"]["action"], "CreateAccessKey")
        self.assertEqual(ev["user"], "arn:aws:iam::1:user/bob")
        self.assertEqual(ev["outcome"], "failure: AccessDenied")
        self.assertEqual(ev["network"]["src_ip"], "203.0.113.5")
        wrapped = normalize({"Records": [{"eventSource": "s3.amazonaws.com", "eventName": "GetObject"}]})
        self.assertEqual(wrapped[0]["event"]["action"], "GetObject")

    def test_suricata_alert(self):
        ev = normalize({"event_type": "alert", "src_ip": "10.0.0.5", "dest_ip": "198.51.100.7",
                        "dest_port": 443, "proto": "TCP",
                        "alert": {"signature": "ET MALWARE beacon", "severity": 1,
                                  "category": "A Network Trojan was detected", "action": "allowed"}})
        self.assertEqual(ev["title"], "ET MALWARE beacon")
        self.assertEqual(ev["kind"], "alert")
        self.assertEqual(ev["network"]["dst_port"], "443")
        self.assertEqual(ev["outcome"], "allowed")

    def test_zeek_dotted_and_nested_ids(self):
        flat = normalize({"ts": 1.0, "uid": "C1", "id.orig_h": "10.0.0.5", "id.resp_h": "8.8.8.8",
                          "id.resp_p": 53, "proto": "udp", "query": "evil.example"})
        nested = normalize({"ts": 1.0, "uid": "C1", "id": {"orig_h": "10.0.0.5", "resp_h": "8.8.8.8",
                                                           "resp_p": 53}, "proto": "udp",
                            "query": "evil.example"})
        self.assertEqual(flat, nested)
        self.assertEqual(flat["source"], "Zeek")
        self.assertIn("evil.example", render(flat))

    def test_okta(self):
        ev = normalize({"eventType": "user.session.start", "displayMessage": "User login to Okta",
                        "actor": {"alternateId": "bob@example.com", "type": "User"},
                        "outcome": {"result": "FAILURE", "reason": "INVALID_CREDENTIALS"},
                        "client": {"ipAddress": "203.0.113.9"}, "severity": "INFO"})
        self.assertEqual(ev["user"], "bob@example.com")
        self.assertEqual(ev["outcome"], "FAILURE: INVALID_CREDENTIALS")
        self.assertEqual(ev["event"]["action"], "user.session.start")

    def test_list_of_events(self):
        txt = render(normalize([{"EventID": 1}, {"EventID": 3}]))
        self.assertTrue(txt.startswith("[1]") and "\n[2]" in txt)


class NoveltyTest(unittest.TestCase):
    TRAIN = [f"Windows event. Event: 4688 (Security). Process: image cmd.exe; user u{i}."
             for i in range(40)]
    CAL = [f"Windows event. Event: 4624 (Security). Process: image svc{i}.exe; user x{i}."
           for i in range(40)]

    def test_familiar_passes_foreign_is_flagged(self):
        g = NoveltyGate.fit(self.TRAIN, self.CAL, alpha=0.05)
        self.assertFalse(g.check("Windows event. Event: 4688 (Security). Process: image cmd.exe; user zz.")["novel"])
        self.assertTrue(g.check("resource \"aws_s3_bucket\" \"b\" { acl = \"public-read\" }")["novel"])

    def test_round_trip(self):
        g = NoveltyGate.fit(self.TRAIN, self.CAL, alpha=0.05)
        with tempfile.TemporaryDirectory() as d:
            g.save(d)
            h = NoveltyGate.load(d)
        probe = "some unrelated text"
        self.assertAlmostEqual(g.p_value(probe), h.p_value(probe), places=3)

    def test_novel_input_abstains_in_decide(self):
        g = NoveltyGate.fit(self.TRAIN, self.CAL, alpha=0.05)
        v = Haruspex(_Fixed("yes", 0.99), novelty=g).decide({**BOOL, "state": "zzzz qqqq"})
        self.assertTrue(v["abstain"])
        self.assertEqual(v["route"], "escalate")


class _Fixed:
    def __init__(self, label, conf):
        self.label, self.conf = label, conf

    def classify_text(self, text, tasks, threshold=0.0, include_confidence=True):
        return {"decision": {"label": self.label, "confidence": self.conf}}


class _ByContent:
    """'yes' for any window containing EVIL, else 'no'; confidence varies by window."""

    def __init__(self):
        self.calls = 0

    def classify_text(self, text, tasks, threshold=0.0, include_confidence=True):
        self.calls += 1
        return {"decision": {"label": "yes" if "EVIL" in text else "no",
                             "confidence": 0.9 + 0.001 * self.calls}}


class WindowTest(unittest.TestCase):
    def test_short_state_is_one_call(self):
        m = _ByContent()
        Haruspex(m).decide({**BOOL, "state": "short"})
        self.assertEqual(m.calls, 1)

    def test_agreeing_windows_take_the_lowest_confidence(self):
        m = _ByContent()
        v = Haruspex(m).decide({**BOOL, "state": "x" * (CHUNK_CHARS * 3)})
        self.assertGreater(m.calls, 1)
        self.assertEqual(v["label"], "no")
        self.assertAlmostEqual(v["confidence"], 0.901, places=5)
        self.assertNotIn("abstain", v)   # no calibrator, no reason: plain verdict

    def test_disagreeing_windows_abstain(self):
        # one window sees the evidence, the others do not -> do not guess
        v = Haruspex(_ByContent()).decide({**BOOL, "state": "x" * (CHUNK_CHARS * 2) + "EVIL"})
        self.assertTrue(v["abstain"])
        self.assertIn("disagree", v["reason"])

    def test_unread_tail_abstains(self):
        v = Haruspex(_ByContent()).decide({**BOOL, "state": "x" * (CHUNK_CHARS * (MAX_WINDOWS + 2))})
        self.assertTrue(v["abstain"])
        self.assertIn("tail", v["reason"])


if __name__ == "__main__":
    unittest.main()
