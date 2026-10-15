"""CVE-applicability resolver: must be correct where confident and NEVER wrong.

The safety property is defer-on-uncertainty — a coverage gap yields None (abstain),
never a wrong verdict. These tests pin the tricky cases that earlier broke (substring
'patched' in 'unpatched', OpenSSL letter versions, Spring4Shell WAR-vs-jar, Drupal
branches) and assert the resolver contradicts no gold label in the shipped dataset.
"""

import json
import unittest
from pathlib import Path

from haruspex.cve import CveResolver, extract_cve, is_applicability
from haruspex.data import gold_label

R = CveResolver()
ROOT = Path(__file__).resolve().parents[1]


def _resolve(cve, state):
    return R.resolve(cve, state)


class TestCveResolver(unittest.TestCase):
    def test_known_true(self):
        self.assertIs(_resolve("CVE-2023-22515", "Confluence 8.3.0"), True)
        self.assertIs(_resolve("CVE-2014-0160", "OpenSSL 1.0.1f on the web server"), True)
        self.assertIs(_resolve("CVE-2019-0708", "Windows 7 SP1 with RDP exposed and unpatched"), True)
        self.assertIs(_resolve("CVE-2017-0144", "Windows 7 with SMBv1 enabled and unpatched"), True)
        self.assertIs(_resolve("CVE-2018-7600", "Drupal 8.4.5"), True)

    def test_known_false(self):
        self.assertIs(_resolve("CVE-2023-22515", "Confluence 7.19.1"), False)
        self.assertIs(_resolve("CVE-2014-0160", "OpenSSL 1.0.2u"), False)
        self.assertIs(_resolve("CVE-2019-0708", "Windows Server 2019 with RDP exposed"), False)
        self.assertIs(_resolve("CVE-2017-0144", "Windows Server 2019 with SMBv1 disabled"), False)
        self.assertIs(_resolve("CVE-2018-7600", "Drupal 8.5.1"), False)
        self.assertIs(_resolve("CVE-2024-4577", "PHP 8.2.10 under PHP-FPM on Linux"), False)
        self.assertIs(_resolve("CVE-2022-22965", "Spring Boot 2.7 as a standalone executable jar with embedded Tomcat"), False)

    def test_defer_on_unknown(self):
        self.assertIsNone(_resolve("CVE-2099-99999", "Some product 1.2.3"))

    def test_patched_not_matched_inside_unpatched(self):
        # 'patched' must not fire on 'unpatched'
        self.assertIsNot(_resolve("CVE-2021-26855", "Exchange Server 2016 unpatched, without the 2021 update"), False)

    def test_never_wrong_on_dataset(self):
        p = ROOT / "data" / "security.jsonl"
        if not p.exists():
            self.skipTest("dataset not present")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        wrong = []
        for r in rows:
            if r.get("category") != "off_cve_applicable" or not is_applicability(r):
                continue
            cve = extract_cve(f"{r.get('question','')} {r.get('state','')}")
            ans = _resolve(cve, str(r.get("state", ""))) if cve else None
            if ans is None:
                continue
            if ("yes" if ans else "no") != gold_label(r):
                wrong.append((cve, r.get("state", "")[:60]))
        self.assertEqual(wrong, [], f"resolver contradicted gold: {wrong}")


if __name__ == "__main__":
    unittest.main()
