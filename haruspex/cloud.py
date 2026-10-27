"""Cloud / IaC scanner-finding routing for Haruspex.

A cloud misconfig scanner (checkov, tfsec, trivy, prowler, …) has ALREADY computed the
structural fact — is Block Public Access on? is the CIDR 0.0.0.0/0? is the port sensitive?
Asking a ~0.5B encoder to re-judge that means negating polarity on templated phrasing
("Ensure BPA enabled" *failed* = BPA absent) where keywords collide with the hardened case
("BPA all true" = safe). The encoder cannot do that reliably, so its errors are confident
and the selective layer cannot catch them — the same situation as CVE applicability.

This module routes a cloud scanner finding to a deterministic classifier instead:
  - a SECURITY control that is violated/absent  -> True  (real, reportable)
  - an OPERATIONAL / best-practice nit          -> False (not reportable alone)
  - a described HARDENED / safe state           -> False
  - anything it is not sure about               -> None  (defer to the model)

The safety property is **defer-on-uncertainty**: a gap returns None (the model decides),
never a wrong verdict. It only fires on findings it recognizes with confidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Marks text as a cloud / IaC finding at all (else this router stays out of the way).
# Substring match (no trailing \b — these appear inside identifiers like aws_s3_bucket);
# short ambiguous tokens keep word boundaries so they don't hit unrelated text.
_CLOUD_RE = re.compile(
    r"(ckv\d?_|ckv_aws|ckv2_|avd-aws|tfsec|checkov|prowler|scoutsuite|terraform|cloudformation|"
    r"aws_|s3[ _]bucket|security[ _]group|iam (policy|role|policies)|\brds\b|\bebs\b|\bkms\b|"
    r"cloudtrail|lambda|\bec2\b|\bvpc\b|\becr\b|\beks\b|api gateway|bucket policy|"
    r"block[ _]public|publicly_accessible|0\.0\.0\.0/0)", re.I)

# A described hardened / safe state (positive polarity) — not a finding.
_HARDENED = [
    r"all four\b.{0,40}\btrue\b",
    r"block[_ ]public[_ ]ac\w*\s*(=|:)?\s*true",
    r"block public access\s+(is\s+)?(fully\s+)?(on|enabled|true)",
    r"block public access\s+enabled",
    r"all\s+(four\s+)?(block[_ ]public|public-access)\w*\s+true",
    r"publicly_accessible\s*(=|:)?\s*false",
    r"not\s+(publicly\s+)?(reachable|accessible)\s+from\s+the\s+internet",
    r"not\s+publicly\s+accessible",
    r"storage_encrypted\s*(=|:)?\s*true",
    r"encryption\s+(is\s+)?enabled",
    r"enable_key_rotation\s*(=|:)?\s*true",
    r"(restricted|scoped)\s+to\s+(a\s+)?(specific|single|one|the)",
    r"from\s+(a\s+)?(single|specific|the)\s+(corporate\s+)?cidr",
    r"security_groups?\s*(=|:)\s*\[",
    r"source_security_group",
    r"imdsv2\s+(is\s+)?required",
    r"\bpass(ed)?\b.*(hardened|secure|no action)",
    r"this is the (hardened|recommended|secure) (state|config)",
    r"cloudtrail\s+(is\s+)?(enabled|on)\b.*multi-region",
    r"\b(not a finding|no action needed|hardened)\b",
]

# A real security violation (negative polarity / exposure).
_VIOLATION = [
    r"0\.0\.0\.0/0.*\b(22|3389|3306|5432|6379|27017|9200|1433|5984|2375|21|23|139|445|2049|5900|11211|all[- ]?ports|any port)\b",
    r"\b(22|3389|3306|5432|6379|27017|9200|1433|5984|2375|21|23|139|445|2049|5900|11211)\b.*0\.0\.0\.0/0",
    r"from_port\s*(=|:)?\s*0\b.*to_port\s*(=|:)?\s*65535",
    r"public[- ]?read|public-read-write|allusers|authenticatedusers",
    r'principal\s*(=|:)?\s*["\']?\*',
    r"block[_ ]public\w*\s*(=|:)?\s*false",
    r"all four\b.{0,40}\bfalse\b",
    r"block public access\s+(is\s+)?(off|disabled|false)",
    r"publicly_accessible\s*(=|:)?\s*true|(?<!not )(?<!n't )publicly accessible|public facing .*no waf|public.*without waf",
    r"storage_encrypted\s*(=|:)?\s*false|not encrypted|unencrypted|encryption\s+(is\s+)?(disabled|not enabled|off)",
    r"(does not|doesn't) have [\w\- ]{0,30}encrypt",
    r'action\s*(=|:)?\s*["\']?\*["\']?\s*(on|,)?\s*.*resource\s*(=|:)?\s*["\']?\*|wildcard (principal|action|resource)',
    r"hardcoded|\bakia[0-9a-z]*\b|(?<!\w)secrets?\b[\w ]{0,20}?\bin (the )?(code|environment|env\b|plaintext)|plaintext (secret|env)",
    r"public (ami|snapshot|image|ecr|repository)|snapshot is public|ami is public",
    r"anonymous (ldap|access|listbucket|getobject|bind)",
    r"(?<!not )(?<!n't )open to the (whole |entire )?internet|open ingress|allows ingress from the internet",
    r"cleartext|plaintext listener|http listener|uses http protocol|imdsv1 (is )?enabled",
    r"no mfa|mfa (is )?(not|disabled)|root account.*access key",
]

# Operational / best-practice / availability nits — not security findings alone.
_OPS = [
    "x-ray", "xray", "tracing", "dead letter", "dead-letter", "dlq", "concurrency",
    "create before destroy", "create-before-destroy", "caching", "cache", "code-sign",
    "code signing", "versioning", "inside a vpc", "in a vpc", "access logging", "flow log",
    "description", "tag", "minor version", "minor upgrade", "auto minor", "container insights",
    "lifecycle", "autoscaling", "auto scaling", "copy tags", "healthcheck", "health check",
    "backup retention", "deletion protection", "multi-az", "multi az", "ebs optimized",
    "ebs-optimized", "latest tag", "image tag", "deployment", "notification", "cross-region replication",
    "point-in-time", "retention period", "performance insights", "enhanced monitoring",
]


# Security-control keywords. In FAILED-check mode a scanner failure on any of these
# means the control is absent/violated -> a real finding, whatever the recommendation
# wording ("Ensure no public access" failed == public access exists).
_SECURITY = [
    "public", "encrypt", "iam", "kms", "secret", "password", "ssl", "tls", " mfa",
    "waf", "wildcard", "privilege", "exfiltration", "permissions management", "anonymous",
    "0.0.0.0", "open access", "back-end", "backend", "root account", "security group",
    "publicly", "exposed", "cross-account", "assume role", "authenticat", "authoriz",
    "no open", "open to", "credential", "access key", "sensitive",
]
# The scanner's own PASSED status: polarity is settled (the control holds), whatever the
# check title or the resource text says. Only status-shaped tokens count — a bare
# "passed" in prose ("traffic passed through the WAF") does not.
_PASSED_RE = re.compile(
    r"^\W*pass(ed)?\b|\bpass(ed)?\s*[:\-]|\b(result|status)\s*[:=]?\s*pass(ed)?\b|"
    r"\b(ckv2?_\w+|avd-\w+-\d+)\s+pass(ed)?\b", re.I)
# A scanner result that represents a FAILED check (a violation), vs a config-state description.
_FAILED_RE = re.compile(
    r"\bfail(ed|s)?\b|\bckv\d?_|\bckv2_|avd-aws|\bviolation\b|does not satisfy|is not (enabled|in place)|"
    r"not configured|should (be|have|not)|ensure (that |the |no |an? )", re.I)


_NEG_RE = re.compile(r"\b(no|not|never|without|isn't|aren't|doesn't|don't)\b")
# The exposure is stated as deliberate: a judgment call for the model, not a rule.
_JUSTIFIED_RE = re.compile(r"\bas intended\b|\bis expected\b|\bby design\b|\bintended (for|as)\b|"
                           r"\bcontains only\b")


def _any(patterns, text: str) -> bool:
    """A pattern matches with no negation in the three words before it, so 'never
    hardcoded' / 'no wildcard principals' / 'not have ... encryption enabled' don't count."""
    for p in patterns:
        for m in re.finditer(p, text):
            before = re.split(r"[.;!?]", text[max(0, m.start() - 60):m.start()])[-1]
            if not _NEG_RE.search(" ".join(before.split()[-3:])):
                return True
    return False


def _kw(keywords, text: str) -> bool:
    """Keyword starts a word: 'tag' hits 'tags' but not 'staged', 'iam' not 'diameter'."""
    return any(re.search(r"(?<!\w)" + re.escape(k.strip()), text) for k in keywords)


def is_cloud_finding(row: dict) -> bool:
    """A boolean row whose text looks like a cloud / IaC scanner finding."""
    if (row.get("type") or "boolean").lower() != "boolean":
        return False
    text = f"{row.get('question','')} {row.get('state','')}"
    return bool(_CLOUD_RE.search(text))


class CloudResolver:
    """Deterministic security-vs-noise classifier for cloud scanner findings.
    Returns True (real) / False (nit or hardened) / None (defer to the model)."""

    def __init__(self, extra_ops: list[str] | None = None):
        self.ops = _OPS + (extra_ops or [])

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "CloudResolver":
        p = Path(model_dir) / "cloud_rules.json"
        if p.exists():   # a corrupt file raises rather than silently dropping its rules
            return cls(extra_ops=json.loads(p.read_text()).get("ops"))
        return cls()

    def classify(self, state: str):
        t = (state or "").lower()
        # 0) the scanner reported PASSED (and not also FAILED): not a finding
        if _PASSED_RE.search(t) and not re.search(r"\bfail(ed|s)?\b", t):
            return False
        # 1) unambiguous exposure/violation wins first (direct phrasing)
        if _any(_VIOLATION, t):
            return None if _JUSTIFIED_RE.search(t) else True
        # 2) FAILED-check mode: a scanner failure resolves polarity for us (the control is
        #    absent), so classify by WHAT control failed — security -> real, ops -> nit.
        if _FAILED_RE.search(t):
            sec, ops = _kw(_SECURITY, t), _kw(self.ops, t)
            if sec != ops:
                return sec
            return None  # failed check, control class unclear or both -> defer
        # 3) a described hardened/safe config state (positive polarity, not a failure)
        if _any(_HARDENED, t):
            return False
        # 4) operational / best-practice nit
        if _kw(self.ops, t):
            return False
        # 5) unsure -> defer to the model
        return None
