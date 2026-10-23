"""Regression tests: haruspex must discriminate assay's actual evidence format.

These tests validate that the model can distinguish real findings from noise
when presented with assay's exact evidence text format:
    {Type} on {URL} param "{Param}" [{sev},{conf},{verified/unverified},tool:{name}]: {evidence}{qualifier}

Tests are expected to FAIL until the model is retrained with the DAST
adjudication data (data/dast_adjudication.jsonl). They serve as the
regression gate: any future model change must pass these.
"""
import os
import pytest

REAL_FINDINGS = [
    {
        "label": "verified-sqli-syntax-error",
        "state": 'SQL Injection on http://target/login param "username" [critical,confirmed,verified-by-replay,tool:sqli-detector]: syntax error near \'OR 1=1-- \' in PostgreSQL error response',
        "min_prob": 0.7,
    },
    {
        "label": "verified-ssti-sentinel",
        "state": 'Server-Side Template Injection (SSTI) on http://target/render param "name" [high,confirmed,verified-by-replay,tool:ssti-detector]: {{7*7}} returns 49 in response, sentinel st12349st123 confirmed Jinja2 engine',
        "min_prob": 0.7,
    },
    {
        "label": "verified-xss-dom",
        "state": 'Cross-Site Scripting (XSS) on http://target/search param "q" [high,confirmed,verified-by-replay,tool:xss-detector]: <script>alert(document.domain)</script> executed in DOM, confirmed by DOM mutation observer',
        "min_prob": 0.7,
    },
    {
        "label": "verified-ssrf-metadata",
        "state": 'Server-Side Request Forgery (SSRF) on http://target/fetch param "url" [critical,confirmed,verified-by-replay,tool:ssrf-detector]: http://169.254.169.254/latest/meta-data/ returns AWS instance metadata',
        "min_prob": 0.7,
    },
    {
        "label": "verified-cmdi-sleep",
        "state": 'Command Injection on http://target/ping param "host" [critical,confirmed,verified-by-replay,tool:cmdi-detector]: ;sleep 5 adds 5.2s delay vs 0.1s baseline, confirmed with id command output uid=33(www-data)',
        "min_prob": 0.7,
    },
]

NOISE_FINDINGS = [
    {
        "label": "css-injection-unverified",
        "state": 'CSS Injection on http://target/search param "q" [medium,low,unverified,tool:cssinj-detector]: background:url(http://evil.com) No impact demonstrated — CSS reflection only.',
        "max_prob": 0.4,
    },
    {
        "label": "hpp-unverified",
        "state": 'HTTP Parameter Pollution on http://target/api/users param "sort" [medium,low,unverified,tool:hpp-detector]: duplicate param accepted Duplicate parameter accepted. No security impact demonstrated.',
        "max_prob": 0.4,
    },
    {
        "label": "ssi-unverified",
        "state": 'SSI Injection on http://target/page param "name" [high,low,unverified,tool:ssi-detector]: <!--#exec cmd="echo test"--> reflected SSI directive reflected but server-side execution not confirmed.',
        "max_prob": 0.4,
    },
    {
        "label": "email-injection-unverified",
        "state": 'Email Header Injection on http://target/contact param "email" [medium,low,unverified,tool:emailinj-detector]: Cc: attacker@evil.com header injected No SMTP interaction observed — header reflection only.',
        "max_prob": 0.4,
    },
    {
        "label": "missing-header-standalone",
        "state": 'Missing header X-Frame-Options on http://target/ param "" [low,low,unverified]: no clickjacking protection on marketing page',
        "max_prob": 0.4,
    },
]

# Novel classes NOT in training data — haruspex must generalize.
NOVEL_FINDINGS = [
    {
        "label": "novel-verified-ldap-injection",
        "state": 'LDAP Injection on http://target/ldap param "uid" [critical,confirmed,verified-by-replay,tool:ldap-detector]: )(uid=*))(|(uid=* returns all LDAP directory entries, baseline returns 1 entry',
        "min_prob": 0.6,
    },
    {
        "label": "novel-noise-autocomplete",
        "state": 'Autocomplete Enabled on http://target/login param "password" [low,low,unverified,tool:form-checker]: password field has autocomplete=on, browsers may cache credentials',
        "max_prob": 0.4,
    },
]

_Q = {
    "type": "boolean",
    "question": "Is this a real, reportable finding?",
    "criteria": {"true": "confirmed exposure with impact", "false": "noise or unexploitable"},
}


def _model_available():
    """Check if a haruspex model is reachable."""
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:3000/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _from_pretrained():
    """Try loading from local runs/sec, fall back to HTTP."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "sec")
    if os.path.isdir(model_dir):
        from haruspex import Haruspex
        return Haruspex.from_pretrained(model_dir)
    return None


@pytest.fixture(scope="module")
def model():
    m = _from_pretrained()
    if m is not None:
        return m
    if _model_available():
        pytest.skip("Model loaded via HTTP — use live server tests instead")
    pytest.skip("No haruspex model available (no runs/sec dir and no server on :3000)")


@pytest.mark.parametrize("case", REAL_FINDINGS, ids=[c["label"] for c in REAL_FINDINGS])
def test_real_finding_scores_high(model, case):
    result = model.decide({**_Q, "state": case["state"]})
    assert result.probability >= case["min_prob"], (
        f"Real finding '{case['label']}' scored {result.probability:.3f}, "
        f"expected >= {case['min_prob']}"
    )


@pytest.mark.parametrize("case", NOISE_FINDINGS, ids=[c["label"] for c in NOISE_FINDINGS])
def test_noise_finding_scores_low(model, case):
    result = model.decide({**_Q, "state": case["state"]})
    assert result.probability <= case["max_prob"], (
        f"Noise finding '{case['label']}' scored {result.probability:.3f}, "
        f"expected <= {case['max_prob']}"
    )


@pytest.mark.parametrize("case", NOVEL_FINDINGS, ids=[c["label"] for c in NOVEL_FINDINGS])
def test_novel_class_generalizes(model, case):
    """Haruspex must handle finding types NOT in training data."""
    result = model.decide({**_Q, "state": case["state"]})
    if "min_prob" in case:
        assert result.probability >= case["min_prob"], (
            f"Novel real finding '{case['label']}' scored {result.probability:.3f}, "
            f"expected >= {case['min_prob']}"
        )
    else:
        assert result.probability <= case["max_prob"], (
            f"Novel noise finding '{case['label']}' scored {result.probability:.3f}, "
            f"expected <= {case['max_prob']}"
        )


def test_separation_gap(model):
    """Real findings must score significantly higher than noise."""
    real_scores = []
    for case in REAL_FINDINGS:
        r = model.decide({**_Q, "state": case["state"]})
        real_scores.append(r.probability)
    noise_scores = []
    for case in NOISE_FINDINGS:
        r = model.decide({**_Q, "state": case["state"]})
        noise_scores.append(r.probability)
    avg_real = sum(real_scores) / len(real_scores)
    avg_noise = sum(noise_scores) / len(noise_scores)
    gap = avg_real - avg_noise
    assert gap >= 0.4, (
        f"Separation gap {gap:.3f} (real avg {avg_real:.3f}, noise avg {avg_noise:.3f}), "
        f"expected >= 0.4"
    )
