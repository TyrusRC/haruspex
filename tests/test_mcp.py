"""MCP tool server: tool functions over a stub-backed router (no model load, no GPU)."""

import asyncio
import subprocess
import sys
import unittest

from mcp.server.mcpserver.exceptions import ToolError

from haruspex.calibrate import SelectiveCalibrator
from haruspex.mcp_server import build_server, make_tools
from haruspex.model import Haruspex
from haruspex.serve import MAX_STATE_CHARS, Router


class _Stub:
    """Stands in for the gliner2 model: returns a fixed classify_text payload."""

    def __init__(self, out):
        self.out = out

    def classify_text(self, text, tasks, threshold=0.0, include_confidence=True):
        return {"decision": self.out}


class _Fixed:
    """A model whose decide() returns a fixed verdict and records the rows it saw."""

    def __init__(self, verdict):
        self.verdict, self.rows = verdict, []

    def decide(self, row):
        self.rows.append(row)
        return dict(self.verdict)


def _router(models: dict, default="web") -> Router:
    r = Router({d: f"stub-{d}" for d in models}, default=default)
    r._cache.update(models)   # pre-loaded: Router.model never calls Haruspex.load
    return r


def _hx(label="yes", conf=0.9, threshold=0.5):
    return Haruspex(_Stub({"label": label, "confidence": conf}),
                    calibrator=SelectiveCalibrator(threshold, 0.1))


class DecideToolTest(unittest.TestCase):
    def test_accepted_verdict_is_actionable(self):
        decide, _ = make_tools(_router({"web": _hx(conf=0.9)}))
        v = decide("SQLi on /login via UNION SELECT", "Is this exploitable?")
        self.assertTrue(v["accept"])
        self.assertFalse(v["escalate"])
        self.assertTrue(v["action"].startswith("ACCEPTED"))
        self.assertEqual((v["domain"], v["domain_source"]), ("web", "default"))

    def test_abstain_says_escalate(self):
        decide, _ = make_tools(_router({"web": _hx(conf=0.55, threshold=0.95)}))
        v = decide("ambiguous finding", "Is this a true positive?")
        self.assertTrue(v["abstain"])
        self.assertTrue(v["escalate"])
        self.assertIn("escalate to a stronger model or a human", v["action"])

    def test_abstain_carries_reason_and_route(self):
        m = _Fixed({"type": "boolean", "label": None, "abstain": True, "accept": False,
                    "route": "external_cve_lookup", "reason": "needs a KEV/NVD lookup"})
        decide, _ = make_tools(_router({"web": m}))
        v = decide("CVE-2024-0001 on nginx 1.2", "Does the CVE apply?")
        self.assertIn("needs a KEV/NVD lookup", v["action"])
        self.assertIn("external_cve_lookup", v["action"])

    def test_uncalibrated_is_not_reported_as_accepted(self):
        decide, _ = make_tools(_router({"web": Haruspex(_Stub({"label": "yes", "confidence": 0.9}))}))
        v = decide("x", "q?")
        self.assertFalse(v["abstain"])
        self.assertTrue(v["action"].startswith("UNCALIBRATED"))

    def test_question_maps_through_shared_row_builder(self):
        m = _Fixed({"type": "choice", "choice": "high", "accept": True, "abstain": False})
        decide, _ = make_tools(_router({"web": m}))
        decide({"finding": "x"}, "Severity?", type="choice", candidates={"high": "h", "low": "l"})
        decide("x", "Rate", type="score", levels=["low", "high"])
        self.assertEqual(m.rows[0], {"type": "choice", "state": {"finding": "x"},
                                     "question": "Severity?", "candidates": {"high": "h", "low": "l"}})
        self.assertEqual(m.rows[1]["levels"], ["low", "high"])

    def test_oversized_state_rejected_before_model(self):
        m = _Fixed({})
        decide, _ = make_tools(_router({"web": m}))
        with self.assertRaises(ToolError):
            decide("a" * (MAX_STATE_CHARS + 1), "q?")
        self.assertEqual(m.rows, [])

    def test_bad_input_is_a_tool_error(self):
        decide, _ = make_tools(_router({"web": _Fixed({})}))
        for kw in ({"type": "choice"}, {"type": "score"}, {"type": "bogus"}):
            with self.assertRaises(ToolError):
                decide("x", "q?", **kw)
        with self.assertRaises(ToolError):
            decide("x", "  ")

    def test_routing_uses_hint_then_inference(self):
        web, mobile = _Fixed({"accept": True, "abstain": False}), _Fixed({"accept": True, "abstain": False})
        decide, _ = make_tools(_router({"web": web, "mobile": mobile}))
        self.assertEqual(decide("plain text", "q?", domain="mobile")["domain_source"], "hint")
        v = decide("android:exported activity in AndroidManifest of the APK", "q?")
        self.assertEqual((v["domain"], v["domain_source"]), ("mobile", "inferred"))
        self.assertEqual(decide("x", "q?", domain="cloud")["domain"], "web")
        self.assertEqual((len(web.rows), len(mobile.rows)), (1, 2))

    def test_list_domains(self):
        _, list_domains = make_tools(_router({"web": _Fixed({}), "mobile": _Fixed({})}))
        d = list_domains()
        self.assertEqual((d["domains"], d["default"]), (["web", "mobile"], "web"))
        self.assertEqual(d["inferred_without_hint"], ["mobile"])


class ServerTest(unittest.TestCase):
    def test_tools_registered_and_callable(self):
        server = build_server(_router({"web": _hx(conf=0.9)}))
        names = {t.name for t in asyncio.run(server.list_tools())}
        self.assertEqual(names, {"decide", "list_domains"})
        res = asyncio.run(server.call_tool("decide", {"state": {"a": 1}, "question": "q?"}))
        self.assertFalse(res.is_error)
        self.assertTrue(res.structured_content["accept"])

    def test_importing_serve_does_not_build_app(self):
        # mcp_server imports serve; with HARUSPEX_ADAPTERS set that must not load a model
        code = ("import sys, haruspex.serve as S; "
                "assert S._app is None and 'starlette' not in sys.modules")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=60, env={"HARUSPEX_ADAPTERS": "web=/nonexistent", "PATH": ""})
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
