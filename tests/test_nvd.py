"""NVD record filter: every kept label must be an NVD-assigned (Primary) CVSS v3 severity."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_nvd import record, windows  # noqa: E402


def _cve(metrics, desc="A flaw allows remote attackers to read files."):
    return {"id": "CVE-2099-0001", "published": "2099-01-01T00:00:00",
            "descriptions": [{"lang": "en", "value": desc}], "metrics": metrics}


def _m(sev, typ):
    return {"type": typ, "cvssData": {"baseSeverity": sev}}


class NvdRecordTest(unittest.TestCase):
    def test_keeps_only_nvd_assigned_scores(self):
        self.assertIsNone(record(_cve({"cvssMetricV31": [_m("HIGH", "Secondary")]})))
        r = record(_cve({"cvssMetricV31": [_m("HIGH", "Secondary"), _m("MEDIUM", "Primary")]}))
        self.assertEqual(r["severity"], "medium")

    def test_falls_back_to_v30_and_skips_none(self):
        self.assertEqual(record(_cve({"cvssMetricV30": [_m("CRITICAL", "Primary")]}))["severity"], "critical")
        self.assertIsNone(record(_cve({"cvssMetricV31": [_m("NONE", "Primary")]})))

    def test_skips_rejected_and_undescribed(self):
        self.assertIsNone(record(_cve({"cvssMetricV31": [_m("LOW", "Primary")]}, desc="** REJECT ** dup")))
        self.assertIsNone(record(_cve({"cvssMetricV31": [_m("LOW", "Primary")]}, desc="")))


class WindowTest(unittest.TestCase):
    def test_windows_are_contiguous_and_within_the_api_cap(self):
        from datetime import date, timedelta
        ws = list(windows(date(2024, 1, 1), date(2024, 12, 31)))
        self.assertEqual(ws[0][0], date(2024, 1, 1))
        self.assertEqual(ws[-1][1], date(2024, 12, 31))
        for (a, b), (c, _) in zip(ws, ws[1:]):
            self.assertEqual(c, b + timedelta(days=1))   # no gap, no overlap
        self.assertTrue(all((b - a).days < 120 for a, b in ws))


if __name__ == "__main__":
    unittest.main()
