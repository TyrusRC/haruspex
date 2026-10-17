"""Serving map, card, and calibrator tests (pure Python; no torch/gliner2 needed)."""

import json
import os
import tempfile
import unittest

from haruspex.calibrate import SelectiveCalibrator
from haruspex.serve import Router, _parse_adapters, _question_to_row


class QuestionMappingTest(unittest.TestCase):
    def test_boolean(self):
        r = _question_to_row("s", {"type": "boolean", "instructions": "q?",
                                   "criteria": {"true": "t", "false": "f"}})
        self.assertEqual(r["type"], "boolean")
        self.assertEqual(r["criteria"], {"true": "t", "false": "f"})

    def test_noul_is_boolean(self):
        # TypeSafe's systemone API calls boolean "noul" — must map to boolean.
        r = _question_to_row("s", {"type": "noul", "instructions": "q?"})
        self.assertEqual(r["type"], "boolean")

    def test_choice(self):
        r = _question_to_row("s", {"type": "choice", "instructions": "q?",
                                   "criteria": {"a": "da", "b": "db"}})
        self.assertEqual(r["type"], "choice")
        self.assertEqual(r["candidates"], {"a": "da", "b": "db"})

    def test_score(self):
        r = _question_to_row("s", {"type": "score", "instructions": "q?",
                                   "criteria": ["low", "high"]})
        self.assertEqual(r["type"], "score")
        self.assertEqual(r["levels"], ["low", "high"])

    def test_state_passthrough(self):
        r = _question_to_row({"a": 1}, {"type": "boolean", "instructions": "q?"})
        self.assertEqual(r["state"], {"a": 1})


class RouterTest(unittest.TestCase):
    def test_parse_adapters(self):
        self.assertEqual(_parse_adapters("web=a,mobile=b"), {"web": "a", "mobile": "b"})
        self.assertEqual(_parse_adapters("web=a;mobile=b"), {"web": "a", "mobile": "b"})
        # a bare path registers as the default domain 'web'
        self.assertEqual(_parse_adapters("runs/sec"), {"web": "runs/sec"})

    def test_resolve_known_and_default(self):
        r = Router({"web": "a", "mobile": "b"}, default="web")
        self.assertEqual(r.resolve("mobile"), "mobile")
        self.assertEqual(r.resolve("web"), "web")
        # unknown / missing domain falls back to default (fail-open, never an error)
        self.assertEqual(r.resolve("cloud"), "web")
        self.assertEqual(r.resolve(None), "web")
        self.assertEqual(r.resolve(""), "web")

    def test_default_falls_back_to_first_registered(self):
        # an unregistered default name must not break routing
        r = Router({"mobile": "b"}, default="web")
        self.assertEqual(r.default, "mobile")
        self.assertEqual(r.resolve("web"), "mobile")

    def test_empty_adapters_rejected(self):
        with self.assertRaises(ValueError):
            Router({})

    def test_smart_switch_infers_mobile_only_without_hint(self):
        r = Router({"web": "a", "mobile": "b"}, default="web")
        # no hint + mobile signal in state -> inferred mobile
        dom, src = r.route(None, "Android app ships AKIA... in res/values/strings.xml inside the APK")
        self.assertEqual((dom, src), ("mobile", "inferred"))
        # no hint + plain web text -> default web (no non-default signal)
        self.assertEqual(r.route(None, "SQL injection confirmed on /login via UNION SELECT"),
                         ("web", "default"))

    def test_explicit_hint_always_wins_over_inference(self):
        r = Router({"web": "a", "mobile": "b"}, default="web")
        # explicit web hint beats mobile-looking text (caller knows its domain)
        self.assertEqual(r.route("web", "android APK debuggable strings.xml"), ("web", "hint"))
        # explicit-but-unregistered -> default, never a guess
        self.assertEqual(r.route("cloud", "android APK debuggable"), ("web", "default"))

    def test_inference_only_over_registered_domains(self):
        # mobile text but no mobile adapter registered -> stays default (web)
        r = Router({"web": "a"}, default="web")
        self.assertEqual(r.route(None, "android APK debuggable strings.xml"), ("web", "default"))


class CardTest(unittest.TestCase):
    def test_card_fills_from_config(self):
        from haruspex.card import build_card
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump({"_name_or_path": "fastino/GLiNER2.5-Decide"}, fh)
        card = build_card(d, "TyrusRC/haruspex")
        self.assertIn("TyrusRC/haruspex", card)
        self.assertIn("fastino/GLiNER2.5-Decide", card)
        self.assertIn("/v1/systemone", card)
        self.assertIn("apache-2.0", card)


class SelectiveCalibratorTest(unittest.TestCase):
    def test_fit_holds_risk_and_abstains(self):
        # High-confidence items are correct, low-confidence ones are wrong:
        # the calibrator should accept the high and abstain on the low.
        pairs = [(0.9, True), (0.85, True), (0.8, True),
                 (0.4, False), (0.45, False), (0.5, False)]
        cal = SelectiveCalibrator.fit(pairs, target_risk=0.1)
        self.assertLessEqual(cal.achieved_risk, 0.1 + 1e-9)
        self.assertTrue(cal.apply(0.95)["accept"])
        self.assertTrue(cal.apply(0.45)["abstain"])

    def test_roundtrip(self):
        cal = SelectiveCalibrator.fit([(0.9, True), (0.2, False)], target_risk=0.1)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "calibrator.json")
        cal.save(p)
        back = SelectiveCalibrator.load(p)
        self.assertEqual(back.threshold, cal.threshold)
        self.assertEqual(back.target_risk, cal.target_risk)

    def test_empty_is_conservative(self):
        cal = SelectiveCalibrator.fit([], target_risk=0.1)
        self.assertEqual(cal.threshold, 1.0)
        self.assertTrue(cal.apply(0.99)["abstain"])   # 0.99 < 1.0 -> abstain
        self.assertTrue(cal.apply(1.0)["accept"])


if __name__ == "__main__":
    unittest.main()


class StateSizeGuardTest(unittest.TestCase):
    """serve.py MAX_STATE_CHARS — the fix for the 2MB-state OOM kill found live."""

    def test_constant_is_sane(self):
        from haruspex.serve import MAX_STATE_CHARS
        self.assertGreater(MAX_STATE_CHARS, 0)
        self.assertLess(MAX_STATE_CHARS, 1_000_000)  # must stay well under OOM territory


class HardeningTest(unittest.TestCase):
    """API-key auth, /metrics, and audit logging — all opt-in, off by default."""

    def _app(self, **env):
        import os
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            import haruspex.serve as S
            orig = S.Router.model
            S.Router.model = lambda self, domain: self._cache.setdefault(
                self.resolve(domain), type("M", (), {"decide": staticmethod(
                    lambda row: {"type": "boolean", "label": "yes", "probability": 1.0})})())
            try:
                return S.build_app({"web": "W"}, default="web")
            finally:
                S.Router.model = orig
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_keyless_by_default(self):
        from starlette.testclient import TestClient
        app = self._app()
        c = TestClient(app)
        r = c.post("/v1/evaluate", json={"state": "x", "questions": {"q": {"type": "boolean", "instructions": "i"}}})
        self.assertEqual(r.status_code, 200)

    def test_api_key_required_when_set(self):
        from starlette.testclient import TestClient
        app = self._app(HARUSPEX_API_KEY="s3cr3t")
        c = TestClient(app)
        body = {"state": "x", "questions": {"q": {"type": "boolean", "instructions": "i"}}}
        self.assertEqual(c.post("/v1/evaluate", json=body).status_code, 401)
        self.assertEqual(c.post("/v1/evaluate", json=body,
                                headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(c.post("/v1/evaluate", json=body,
                                headers={"Authorization": "Bearer s3cr3t"}).status_code, 200)

    def test_metrics_endpoint(self):
        from starlette.testclient import TestClient
        app = self._app()
        c = TestClient(app)
        c.post("/v1/evaluate", json={"state": "x", "questions": {"q": {"type": "boolean", "instructions": "i"}}})
        r = c.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("haruspex_requests_total", r.text)

    def test_audit_log_written(self):
        import json as _json
        import tempfile, os as _os
        from starlette.testclient import TestClient
        fd, path = tempfile.mkstemp()
        _os.close(fd)
        try:
            app = self._app(HARUSPEX_AUDIT_LOG=path)
            c = TestClient(app)
            c.post("/v1/evaluate", json={"state": "x", "questions": {"q": {"type": "boolean", "instructions": "i"}}})
            with open(path) as f:
                lines = [_json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["domain"], "web")
            self.assertNotIn("state", lines[0])  # raw state must never be logged
        finally:
            _os.unlink(path)
