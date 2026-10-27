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
        # 200 high-confidence items, all correct; 200 low ones, all wrong: enough
        # evidence to certify the high band and abstain on the low one.
        pairs = [(0.9, True)] * 200 + [(0.45, False)] * 200
        cal = SelectiveCalibrator.fit(pairs, target_risk=0.1)
        self.assertLessEqual(cal.achieved_risk, 0.1 + 1e-9)
        self.assertTrue(cal.apply(0.95)["accept"])
        self.assertTrue(cal.apply(0.45)["abstain"])

    def test_too_little_evidence_abstains(self):
        # 3 correct items cannot certify a 10% risk at confidence 1-delta
        cal = SelectiveCalibrator.fit([(0.9, True), (0.85, True), (0.8, True)], target_risk=0.1)
        self.assertTrue(cal.apply(0.99)["abstain"])

    def test_risk_guarantee_holds_out_of_sample(self):
        # Confidence c ~ U(0.5, 1), correct with probability c, so the TRUE risk among
        # accepted items at threshold t is (1-t)/2 (t >= 0.5) and 0.25 at t = 0.
        # Over many calibration draws, P(true risk > target) must stay <= delta.
        import random
        rng = random.Random(0)
        alpha, delta, trials, bad = 0.1, 0.1, 200, 0
        for _ in range(trials):
            pairs = []
            for _ in range(300):
                c = rng.uniform(0.5, 1.0)
                pairs.append((c, rng.random() < c))
            t = SelectiveCalibrator.fit(pairs, target_risk=alpha, delta=delta).threshold
            if t is not None and (0.25 if t < 0.5 else (1 - t) / 2) > alpha:
                bad += 1
        self.assertLessEqual(bad / trials, delta)

    def test_per_type_thresholds(self):
        # boolean is wrong at 0.6, choice is right at 0.6: one shared cutoff can't serve both
        pairs = ([(0.9, True, "boolean")] * 200 + [(0.6, False, "boolean")] * 200 +
                 [(0.6, True, "choice")] * 200)
        cal = SelectiveCalibrator.fit(pairs, target_risk=0.1)
        self.assertTrue(cal.apply(0.6, "choice")["accept"])
        self.assertTrue(cal.apply(0.6, "boolean")["abstain"])
        self.assertTrue(cal.apply(0.99, "score")["abstain"])   # never calibrated -> abstain

    def test_roundtrip(self):
        cal = SelectiveCalibrator.fit([(0.9, True, "boolean")] * 100, target_risk=0.1)
        cal.temperature = 1.7
        d = tempfile.mkdtemp()
        p = os.path.join(d, "calibrator.json")
        cal.save(p)
        back = SelectiveCalibrator.load(p)
        self.assertEqual(back.to_dict(), cal.to_dict())

    def test_legacy_file_still_loads(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "calibrator.json")
        with open(p, "w") as fh:
            json.dump({"threshold": 0.7, "target_risk": 0.2}, fh)
        cal = SelectiveCalibrator.load(p)
        self.assertTrue(cal.apply(0.8, "choice")["accept"])
        self.assertTrue(cal.apply(0.6, "boolean")["abstain"])

    def test_empty_fails_closed(self):
        cal = SelectiveCalibrator.fit([], target_risk=0.1)
        self.assertIsNone(cal.threshold)
        self.assertTrue(cal.apply(1.0)["abstain"])

    def test_temperature_fit_softens_overconfidence(self):
        from haruspex.calibrate import fit_temperature
        # says 0.99 but is right only 70% of the time -> T > 1 pulls it toward 0.5
        pairs = [(0.99, True)] * 70 + [(0.99, False)] * 30
        T = fit_temperature(pairs)
        self.assertGreater(T, 1.0)
        cal = SelectiveCalibrator(None, 0.1, temperature=T)
        self.assertAlmostEqual(cal.scale(0.99), 0.7, delta=0.03)


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


class InputValidationTest(unittest.TestCase):
    """Malformed or oversized requests get a clean 4xx, never a 500 or an OOM."""

    _app = HardeningTest._app

    def setUp(self):
        from starlette.testclient import TestClient
        self.c = TestClient(self._app(), raise_server_exceptions=False)

    def test_malformed_json_is_400(self):
        r = self.c.post("/v1/evaluate", content=b"{not json", headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)

    def test_non_object_body_is_400(self):
        self.assertEqual(self.c.post("/v1/evaluate", json=[1, 2]).status_code, 400)
        self.assertEqual(self.c.post("/v1/evaluate", json={"state": "x", "questions": [1]}).status_code, 400)

    def test_non_dict_question_is_a_per_question_error(self):
        r = self.c.post("/v1/evaluate", json={"state": "x", "questions": {"q": "boolean?"}})
        self.assertEqual(r.status_code, 200)
        self.assertIn("error", r.json()["answers"]["q"])

    def test_oversized_body_rejected_before_parsing(self):
        from haruspex.serve import MAX_BODY_BYTES
        big = b'{"state": "' + b"a" * (MAX_BODY_BYTES + 1) + b'", "questions": {}}'
        self.assertEqual(self.c.post("/v1/evaluate", content=big).status_code, 413)

    def test_too_many_questions_rejected(self):
        from haruspex.serve import MAX_QUESTIONS
        qs = {f"q{i}": {"type": "boolean", "instructions": "i"} for i in range(MAX_QUESTIONS + 1)}
        self.assertEqual(self.c.post("/v1/evaluate", json={"state": "x", "questions": qs}).status_code, 413)


class RouterConcurrencyTest(unittest.TestCase):
    def test_concurrent_first_requests_load_once(self):
        import threading
        import time as _time
        import haruspex.model as M
        loads = []

        def slow_load(*a, **k):
            loads.append(1)
            _time.sleep(0.05)
            return object()

        orig = M.Haruspex.load
        M.Haruspex.load = staticmethod(slow_load)
        try:
            r = Router({"web": "a"})
            ts = [threading.Thread(target=r.model, args=("web",)) for _ in range(8)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        finally:
            M.Haruspex.load = orig
        self.assertEqual(len(loads), 1)


class TLSArgsTest(unittest.TestCase):
    """--ssl-keyfile/--ssl-certfile must be given together or rejected (main()'s CLI guard)."""

    def test_mismatched_ssl_args_rejected(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "haruspex.serve", "--model", "x",
                            "--ssl-keyfile", "/tmp/nonexistent-key.pem"],
                           capture_output=True, text=True, timeout=15)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ssl-keyfile and --ssl-certfile", r.stderr + r.stdout)
