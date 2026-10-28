"""OTRF converter: labels need two agreeing annotations; only documented-command process
trees count as attacker events; splits are per dataset."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_otrf import attacker_events, is_seed, label, split_of, tokens, windows  # noqa: E402

SYSMON = "Microsoft-Windows-Sysmon/Operational"


def _meta(folder, tactics):
    return {"files": [{"link": f"https://x/master/datasets/atomic/windows/{folder}/host/a.zip"}],
            "attack_mappings": [{"technique": "T1003", "tactics": tactics}]}


def _proc(guid, parent, image, cmd, pid, ts):
    return {"Channel": SYSMON, "EventID": 1, "ProcessGuid": guid, "ParentProcessGuid": parent,
            "Image": image, "CommandLine": cmd, "ProcessId": pid, "Hostname": "WS5.lab",
            "@timestamp": ts}


def _load(guid, dll, ts):
    return {"Channel": SYSMON, "EventID": 7, "ProcessGuid": guid, "ImageLoaded": dll,
            "Hostname": "WS5", "@timestamp": ts}


VIEW = tokens(r"C:\Users\x>esentutl.exe /y /vss %SystemRoot%/system32/config/SAM /d C:\ProgramData\SAM")
EVENTS = [
    _proc("{P}", "{X}", r"C:\Windows\System32\cmd.exe", "cmd.exe", 100, "t1"),
    _proc("{S}", "{P}", r"C:\Windows\System32\esentutl.exe",
          r"esentutl.exe  /y /vss C:\windows/system32/config/SAM /d C:\ProgramData\SAM", 200, "t2"),
    _proc("{C}", "{S}", r"C:\Windows\System32\conhost.exe", "conhost.exe 0xffffffff", 300, "t3"),
    _proc("{B}", "{X}", r"C:\Windows\System32\svchost.exe", "svchost.exe -k netsvcs", 400, "t3"),
    _load("{S}", r"C:\Windows\System32\ntdll.dll", "t4"),     # also loaded by background
    _load("{B}", r"C:\Windows\System32\ntdll.dll", "t4"),
    _load("{S}", r"C:\Windows\System32\vssapi.dll", "t5"),    # attack-specific
    {"Channel": "Security", "EventID": 4663, "ProcessId": "0xc8", "Hostname": "WS5.lab",
     "ObjectName": r"C:\Windows\System32\config\SAM", "@timestamp": "t6"},
    {"Channel": "Security", "EventID": 4663, "ProcessId": "0xc8", "Hostname": "WS5.lab",
     "ObjectName": "before-start", "@timestamp": "t0"},       # pid 200 not yet the seed
]


class LabelTest(unittest.TestCase):
    def test_folder_must_be_confirmed_by_mappings(self):
        self.assertEqual(label(_meta("credential_access", ["TA0006"])), "credential_access")
        self.assertIsNone(label(_meta("credential_access", ["TA0004", "TA0005"])))
        self.assertIsNone(label(_meta("other", ["TA0005"])))


class SeedTest(unittest.TestCase):
    def test_documented_command_is_seed_background_is_not(self):
        self.assertTrue(is_seed(r"C:\Windows\System32\esentutl.exe",
                                r"esentutl.exe /y /vss C:\windows/system32/config/SAM /d C:\ProgramData\SAM", VIEW))
        self.assertFalse(is_seed(r"C:\Windows\System32\esentutl.exe", "esentutl.exe /mh other.db /q", VIEW))
        self.assertFalse(is_seed(r"C:\Windows\System32\svchost.exe", "svchost.exe /y /vss", VIEW))
        self.assertFalse(is_seed(r"C:\Windows\System32\esentutl.exe", "esentutl.exe", VIEW))


class AttributionTest(unittest.TestCase):
    def setUp(self):
        self.got = attacker_events(EVENTS, VIEW)

    def test_seed_and_descendants_only(self):
        guids = {e.get("ProcessGuid") for e in self.got if e["EventID"] == 1}
        self.assertEqual(guids, {"{S}", "{C}"})           # parent cmd and svchost excluded

    def test_commonplace_events_dropped(self):
        dlls = {e.get("ImageLoaded") for e in self.got if e["EventID"] == 7}
        self.assertEqual(dlls, {r"C:\Windows\System32\vssapi.dll"})

    def test_security_events_by_pid_after_start(self):
        objs = [e["ObjectName"] for e in self.got if e["EventID"] == 4663]
        self.assertEqual(objs, [r"C:\Windows\System32\config\SAM"])

    def test_reused_pid_is_not_linked(self):
        reuse = EVENTS + [_proc("{R}", "{X}", r"C:\Windows\System32\notepad.exe", "notepad.exe", 200, "t7")]
        self.assertFalse([e for e in attacker_events(reuse, VIEW) if e["EventID"] == 4663])


class RenderSplitTest(unittest.TestCase):
    def test_windows_redact_lab_identifiers(self):
        ev = dict(EVENTS[1], User="LAB\\wardog", CommandLine="esentutl.exe /y C:\\T1003.002.db")
        text = windows([ev])[0]
        self.assertNotIn("wardog", text)
        self.assertNotIn("WS5", text)
        self.assertNotIn("T1003", text)
        self.assertNotIn("Event: 1 1", windows([dict(ev, Task="1")])[0])   # numeric Task code

    def test_split_is_per_dataset_and_stable(self):
        self.assertEqual(split_of("SDWIN-1", 0), split_of("SDWIN-1", 0))
        self.assertEqual({split_of(f"SDWIN-{i}", 0) for i in range(200)}, {"train", "dev", "cal", "test"})


if __name__ == "__main__":
    unittest.main()
