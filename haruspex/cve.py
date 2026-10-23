"""CVE-applicability routing for Haruspex.

Fine-grained "is <product@version> affected by CVE-X?" is a *knowledge lookup*, not a
judgment the ~0.5B encoder can do reliably — its errors there are confident, so the
selective layer cannot catch them. This module routes applicability questions to a
deterministic resolver instead. The safety property is **defer-on-uncertainty**: the
resolver returns a verdict ONLY when it is confident, otherwise `None` → the decision
layer abstains and defers to an external KEV/NVD lookup. A coverage gap can therefore
cause an *abstention*, never a false positive.

Callers may inject their own resolver (e.g. praetor's `lookup_cve`) via
`Haruspex(..., cve_resolver=fn)` where `fn(cve_id, text) -> True|False|None`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
_VER_RE = re.compile(r"\d+(?:[.\-]\d+)+[a-z]?")
# "is ... affected / applicable / vulnerable to CVE" phrasing marks an applicability Q
_APPLIC_RE = re.compile(r"affect|applicab|vulnerable", re.I)


def extract_cve(text: str) -> str | None:
    m = CVE_RE.search(text or "")
    return m.group(0).upper() if m else None


def is_applicability(row: dict) -> bool:
    """A boolean 'is the target affected by CVE-x' question."""
    if (row.get("type") or "boolean").lower() != "boolean":
        return False
    q = f"{row.get('question','')} {row.get('state','')}"
    return bool(extract_cve(q)) and bool(_APPLIC_RE.search(row.get("question", "")))


def _vt(s: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", s)[:5])


def _has(text: str, kw: str) -> bool:
    """Match a keyword in text. Version-like keywords (containing a digit) match as a
    substring so '1.0.1' hits '1.0.1f'; word keywords match on word boundaries so
    'patched' does NOT match inside 'unpatched'."""
    kw = kw.lower()
    if any(c.isdigit() for c in kw):
        return kw in text
    return re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text) is not None


# Seed ranges for CVEs the security pack uses. Each entry is conservative; anything not
# encoded (or a version that cannot be parsed) returns None → defer. Keys:
#   products     : target text must contain one of these (else not applicable → False)
#   requires_all : every keyword must be present (else not applicable → False)
#   includes     : if any present → affected (True)
#   excludes     : if any present → not affected (False)
#   introduced   : affected only at/above this version
#   fixed_in     : not affected at/above this version
_SEED: dict[str, dict] = {
    "CVE-2023-22515": {"products": ["confluence"], "introduced": "8.0.0", "fixed_in": "8.5.2"},
    "CVE-2021-26084": {"products": ["confluence"], "fixed_in": "7.13.0"},
    "CVE-2022-42889": {"products": ["commons", "text4shell"], "introduced": "1.5", "fixed_in": "1.10"},
    "CVE-2022-22965": {"products": ["spring"], "fixed_in": "5.3.18",
                        "excludes": ["standalone", "executable jar", "jar"]},
    "CVE-2021-44228": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.15"},
    "CVE-2021-43798": {"products": ["grafana"], "introduced": "8.0", "fixed_in": "8.3.1"},
    "CVE-2022-3602":  {"products": ["openssl"], "introduced": "3.0.0", "fixed_in": "3.0.7"},
    "CVE-2022-0847":  {"products": ["kernel", "linux"], "introduced": "5.8", "fixed_in": "5.16.11"},
    "CVE-2023-7028":  {"products": ["gitlab"], "introduced": "16.1", "fixed_in": "16.7.2"},
    "CVE-2017-5638":  {"products": ["struts"], "fixed_in": "2.5.11"},
    "CVE-2024-23897": {"products": ["jenkins"], "fixed_in": "2.442"},
    "CVE-2024-4577":  {"products": ["php"], "requires_all": ["windows"], "fixed_in": "8.3.8"},
    "CVE-2021-40438": {"products": ["apache", "httpd"], "fixed_in": "2.4.49"},
    "CVE-2023-4966":  {"products": ["netscaler", "citrix"], "fixed_in": "13.1-49.15"},
    "CVE-2014-0160":  {"products": ["openssl"], "includes": ["1.0.1"], "excludes": ["1.0.2", "1.1.", "3.0", "1.0.0"]},
    "CVE-2019-0708":  {"excludes": ["2019", "2016", "2012", "windows 10", "windows 11"],
                        "includes": ["windows 7", "server 2008", "server 2003", "windows xp", "vista"]},
    "CVE-2017-0144":  {"excludes": ["disabled", "server 2019", "server 2016", "server 2012"],
                        "includes": ["smbv1 enabled", "unpatched"]},
    "CVE-2021-26855": {"products": ["exchange"], "excludes": ["fully patched", "patched"],
                        "includes": ["without", "unpatched", "pre-2021"]},
    "CVE-2016-5195":  {"products": ["kernel", "linux"], "fixed_in": "4.8.3"},
    "CVE-2014-6271":  {"products": ["bash"], "excludes": ["windows", "no gnu bash", "no bash"]},
    "CVE-2020-14882": {"products": ["weblogic"]},
    "CVE-2018-7600":  {"products": ["drupal"],
                        "branches": [{"prefix": "8.", "fixed_in": "8.5.1"},
                                     {"prefix": "7.", "fixed_in": "7.58"}]},
    # --- Web servers ---
    "CVE-2020-1938":  {"products": ["tomcat"],
                        "branches": [{"prefix": "9.", "fixed_in": "9.0.31"},
                                     {"prefix": "8.", "fixed_in": "8.5.51"},
                                     {"prefix": "7.", "fixed_in": "7.0.100"}]},
    "CVE-2021-23017": {"products": ["nginx"], "introduced": "0.6.18", "fixed_in": "1.21.0"},
    "CVE-2017-7269":  {"products": ["iis"], "includes": ["6.0"],
                        "excludes": ["7.0", "7.5", "8.0", "8.5", "10.0"]},
    "CVE-2021-41773": {"products": ["apache", "httpd"], "introduced": "2.4.49", "fixed_in": "2.4.50"},
    "CVE-2021-42013": {"products": ["apache", "httpd"], "introduced": "2.4.49", "fixed_in": "2.4.51"},
    # --- CMS ---
    "CVE-2023-32243": {"products": ["wordpress", "elementor"], "introduced": "5.4.0", "fixed_in": "5.7.2"},
    "CVE-2023-23752": {"products": ["joomla"], "introduced": "4.0.0", "fixed_in": "4.2.8"},
    "CVE-2022-24086": {"products": ["magento", "adobe commerce"], "fixed_in": "2.4.3-p2"},
    "CVE-2019-11043": {"products": ["php", "fpm", "php-fpm", "nginx"],
                        "requires_all": ["nginx"], "introduced": "7.1.0", "fixed_in": "7.3.11"},
    # --- Java ecosystem ---
    "CVE-2021-45046": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.16"},
    "CVE-2021-45105": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.17"},
    "CVE-2022-22963": {"products": ["spring", "cloud"], "fixed_in": "3.2.2"},
    "CVE-2017-12149": {"products": ["jboss"], "includes": ["5.", "6."],
                        "excludes": ["7.", "eap 7"]},
    "CVE-2020-2555":  {"products": ["weblogic", "coherence"], "fixed_in": "12.2.1.4.0"},
    "CVE-2023-21931": {"products": ["weblogic"], "fixed_in": "14.1.1.0.0"},
    # --- Network appliances ---
    "CVE-2024-21762": {"products": ["fortios", "fortigate"], "introduced": "7.2.0", "fixed_in": "7.4.3"},
    "CVE-2022-40684": {"products": ["fortios", "fortigate", "fortiproxy"],
                        "introduced": "7.0.0", "fixed_in": "7.2.2"},
    "CVE-2023-20198": {"products": ["cisco", "ios xe"], "introduced": "16.12", "fixed_in": "17.9.4a"},
    "CVE-2024-3400":  {"products": ["palo alto", "pan-os"], "introduced": "10.2.0", "fixed_in": "10.2.9-h1"},
    "CVE-2023-46805": {"products": ["ivanti", "pulse"], "fixed_in": "22.6R2.2"},
    "CVE-2024-21887": {"products": ["ivanti", "pulse"], "fixed_in": "22.6R2.2"},
    # --- Mail ---
    "CVE-2021-34473": {"products": ["exchange"], "excludes": ["fully patched", "cu23", "cu22"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2021-34523": {"products": ["exchange"], "excludes": ["fully patched"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2021-31207": {"products": ["exchange"], "excludes": ["fully patched"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2022-27925": {"products": ["zimbra"], "fixed_in": "8.8.15p31.1"},
    "CVE-2022-41040": {"products": ["exchange"],
                        "includes": ["2019", "2016"], "excludes": ["fully patched"]},
    # --- Container / K8s ---
    "CVE-2024-21626": {"products": ["runc", "containerd", "docker"], "fixed_in": "1.1.12"},
    "CVE-2022-0185":  {"products": ["kernel", "linux"], "introduced": "5.1", "fixed_in": "5.16.2"},
    "CVE-2022-0811":  {"products": ["cri-o"], "introduced": "1.19", "fixed_in": "1.23.2"},
    # --- Database ---
    "CVE-2023-39417": {"products": ["postgresql"], "introduced": "12.0", "fixed_in": "15.4"},
    "CVE-2023-22809": {"products": ["sudo"], "introduced": "1.8.0", "fixed_in": "1.9.12p2"},
    # --- VPN / Remote access ---
    "CVE-2018-13379": {"products": ["fortios", "fortigate"], "introduced": "5.6.3", "fixed_in": "6.0.5"},
    "CVE-2019-19781": {"products": ["citrix", "netscaler", "adc"], "fixed_in": "13.0-47.24"},
    "CVE-2023-27997": {"products": ["fortios", "fortigate"], "introduced": "6.0.0", "fixed_in": "7.2.5"},
    # --- Misc high-profile ---
    "CVE-2023-44487": {"products": ["http2", "http/2", "h2"],
                        "excludes": ["patched", "mitigated"]},
    "CVE-2024-6387":  {"products": ["openssh"], "introduced": "8.5", "fixed_in": "9.8",
                        "excludes": ["9.8p1", "9.8p2", "9.9", "10."]},
    "CVE-2023-38408": {"products": ["openssh"], "fixed_in": "9.4"},
    "CVE-2022-22947": {"products": ["spring", "cloud", "gateway"], "introduced": "3.1.0", "fixed_in": "3.1.1"},
}


class CveResolver:
    def __init__(self, ranges: dict | None = None):
        self.ranges = ranges if ranges is not None else _SEED

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "CveResolver | None":
        p = Path(model_dir) / "cve_ranges.json"
        if p.exists():
            try:
                return cls(json.loads(p.read_text()))
            except (ValueError, OSError):
                return None
        return cls()  # built-in seed

    def resolve(self, cve: str, text: str):
        """Return True/False if confident, else None (defer)."""
        e = self.ranges.get(cve.upper())
        if not e:
            return None
        t = (text or "").lower()
        prods = e.get("products")
        if prods and not any(_has(t, p) for p in prods):
            return False  # question's CVE is for a product the target isn't running
        for kw in e.get("requires_all", []):
            if not _has(t, kw):
                return False  # a required precondition (e.g. Windows CGI) is absent
        for x in e.get("excludes", []):
            if _has(t, x):
                return False
        inc = e.get("includes")
        if inc is not None and any(_has(t, i) for i in inc):
            return True
        # per-branch version rules (products with multiple maintained lines, e.g. Drupal)
        fixed, intro = e.get("fixed_in"), e.get("introduced")
        vers = _VER_RE.findall(t)
        for br in e.get("branches", []):
            bv = next((x for x in vers if x.startswith(br["prefix"])), None)
            if bv is None:
                continue
            v = _vt(bv)
            if br.get("introduced") and v < _vt(br["introduced"]):
                return False
            if br.get("fixed_in") and v >= _vt(br["fixed_in"]):
                return False
            return True
        if "branches" in e:
            return None  # a branch model but no matching version parsed → defer
        if fixed or intro:
            if not vers:
                return None
            v = _vt(vers[0])
            if intro and v < _vt(intro):
                return False
            if fixed and v >= _vt(fixed):
                return False
            return True
        if inc is not None:
            return False  # had an includes-set, matched none, no version rule
        return None  # product matched but nothing decisive → defer
