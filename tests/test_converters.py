"""Split / filter logic of scripts/convert_nvd.py and scripts/convert_owasp_benchmark.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_nvd as nvd  # noqa: E402
import convert_owasp_benchmark as owasp  # noqa: E402


def _nvd_file(recs, extra_lines=()):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in recs:
        fh.write(json.dumps(r) + "\n")
    for line in extra_lines:
        fh.write(line + "\n")
    fh.close()
    return Path(fh.name)


def _cve(i, published, sev="high", desc=None):
    return {"id": f"CVE-2000-{i:04d}", "published": published,
            "description": desc or f"flaw number {i}", "severity": sev}


class NvdTest(unittest.TestCase):
    def test_load_drops_duplicates_bad_labels_and_partial_lines(self):
        p = _nvd_file([_cve(1, "2020-01-01T00:00:00.000"), _cve(1, "2020-01-01T00:00:00.000"),
                       _cve(2, "2020-01-01T00:00:00.000", sev="none"),
                       _cve(3, "2020-01-01T00:00:00.000", desc=" ")],
                      extra_lines=['{"id": "CVE-2000-0004", "publ'])
        self.addCleanup(p.unlink)
        self.assertEqual([r["id"] for r in nvd.load(p)], ["CVE-2000-0001"])

    def test_newest_go_to_test_and_shared_descriptions_leave_train(self):
        old = [_cve(i, f"2015-01-{1 + i % 28:02d}T00:00:00") for i in range(200)]
        new = [_cve(900, "2024-06-01T00:00:00"), _cve(901, "2023-07-01T00:00:00")]
        dup = _cve(902, "2016-01-01T00:00:00", desc=new[0]["description"])
        parts = nvd.split(old + new + [dup], test_days=365, seed=0)
        self.assertEqual({r["id"] for r in parts["test"]}, {"CVE-2000-0900", "CVE-2000-0901"})
        rest = [r["id"] for s in ("train", "dev", "cal") for r in parts[s]]
        self.assertNotIn("CVE-2000-0902", rest)              # its text is in test
        self.assertEqual(sorted(rest), sorted(r["id"] for r in old))   # disjoint, complete
        self.assertTrue(parts["dev"] and parts["cal"] and parts["train"])

    def test_identical_descriptions_share_a_split(self):
        recs = [_cve(i, "2015-01-01T00:00:00", desc=f"template {i % 5}") for i in range(50)]
        recs.append(_cve(99, "2024-01-01T00:00:00", desc="newest"))
        parts = nvd.split(recs, 365, seed=3)
        for d in {r["description"] for r in recs[:-1]}:
            homes = {s for s, rs in parts.items() for r in rs if r["description"] == d}
            self.assertEqual(len(homes), 1, d)

    def test_balanced_caps_per_level_and_targets_index(self):
        recs = [_cve(i, "2015-01-01T00:00:00", sev=nvd.LEVELS[i % 2]) for i in range(30)]
        rows = nvd.balanced(recs, per_level=4, seed=0, split_name="train")
        self.assertEqual(sorted(r["target"] for r in rows), [0] * 4 + [1] * 4)
        self.assertEqual(rows[0]["levels"], nvd.LEVELS)


JAVA = '''/** license header, GPL */
package x;
import y;
public class BenchmarkTest00001 extends HttpServlet {
    @Override
    public void doGet(HttpServletRequest request) { doPost(request); }

    @Override
    public void doPost(HttpServletRequest request) {
        String a = "http://host/path"; // condition 'B', which is safe
        /* this whole flow is 'safe' */
        char c = '/';
        String alg = benchmarkprops.getProperty("hashAlg1", "SHA512");
    }
}
'''


class OwaspTest(unittest.TestCase):
    def test_state_keeps_doPost_only_and_drops_comments(self):
        s = owasp.extract_state(JAVA, {"hashAlg1": "MD5", "unused": "x"})
        self.assertTrue(s.startswith("public void doPost"))
        for gone in ("doGet", "license", "import", "safe", "/*"):
            self.assertNotIn(gone, s)
        self.assertIn('"http://host/path";', s)               # literal survives
        self.assertIn("char c = '/';", s)
        self.assertTrue(s.endswith("benchmark.properties:\nhashAlg1=MD5"))

    def test_expected_csv_skips_header_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.csv"
            p.write_text("# test name, category, real vulnerability, cwe\n"
                         "BenchmarkTest00001,pathtraver,true,22\n"
                         "BenchmarkTest00002,hash,false,328\n")
            self.assertEqual(owasp.read_expected(p),
                             [("BenchmarkTest00001", "pathtraver", True, 22),
                              ("BenchmarkTest00002", "hash", False, 328)])
            p.write_text("BenchmarkTest00003,unknowncat,true,1\n")
            with self.assertRaises(SystemExit):
                owasp.read_expected(p)

    def test_refuses_output_inside_repo(self):
        argv = sys.argv
        sys.argv = ["x", "--src", "/nonexistent", "--out", str(owasp.REPO / "data" / "o.jsonl")]
        try:
            with self.assertRaises(SystemExit) as cm:
                owasp.main()
            self.assertIn("inside the repo", str(cm.exception))
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
