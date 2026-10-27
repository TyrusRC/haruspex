"""CVE-applicability routing for Haruspex.

Fine-grained "is <product@version> affected by CVE-X?" is a *knowledge lookup*, not a
judgment the ~0.5B encoder can do reliably — its errors there are confident, so the
selective layer cannot catch them. This module routes applicability questions to a
deterministic resolver instead. The safety property is **defer-on-uncertainty**: the
resolver returns a verdict ONLY when it is confident, otherwise `None` → the decision
layer abstains and defers to an external KEV/NVD lookup. A coverage gap can therefore
cause an *abstention*, never a wrong verdict in either direction: a product or version
that is not stated is unknown, not "not affected".

Callers may inject their own resolver (e.g. praetor's `lookup_cve`) via
`Haruspex(..., cve_resolver=fn)` where `fn(cve_id, text) -> True|False|None`.

NOTE (ceiling): versions are parsed from free text and compared with a generic
numeric+letter ordering, not per-ecosystem rules (pre-releases like `2.0-beta9` sort
wrong). The upgrade path is structured input (purl + version) checked against OSV
ranges with an ecosystem-aware comparator such as `univers`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
# A version token in lowercased text: 2.4.49, 1.0.1g, 9.8p1, 13.1-49.15, 22.6r2.2.
_VER = r"\d+(?:\.\d+)+(?:[a-z]+\d*)?(?:[.\-]\d+[a-z]*\d*)*"
# "is ... affected / applicable / vulnerable to CVE" phrasing marks an applicability Q
_APPLIC_RE = re.compile(r"affect|applicab|vulnerable", re.I)
# Distro packages backport fixes without bumping the upstream version, so an upstream
# range cannot decide them — defer.
_DISTRO_RE = re.compile(r"\b(ubuntu|debian|deb\d+u\d+|rhel|red hat|centos|rocky|almalinux|"
                        r"suse|sles|amazon linux|alpine|el[6-9])\b")


def extract_cve(text: str) -> str | None:
    m = CVE_RE.search(text or "")
    return m.group(0).upper() if m else None


def is_applicability(row: dict) -> bool:
    """A boolean 'is the target affected by CVE-x' question."""
    if (row.get("type") or "boolean").lower() != "boolean":
        return False
    q = f"{row.get('question','')} {row.get('state','')}"
    return bool(extract_cve(q)) and bool(_APPLIC_RE.search(row.get("question", "")))


def _vkey(s: str, strip: bool = True) -> tuple:
    """Order key: '1.0.1g' -> ((1,''),(0,''),(1,'g')), so 1.0.1 < 1.0.1f < 1.0.1g and
    9.8 < 9.8p1. Whitespace is dropped so '8.8.15 P31' keys like '8.8.15p31'. Trailing
    zero components are stripped so 2.16 == 2.16.0 (not for prefixes: '6.0' != '6')."""
    k = [(int(n), a) for n, a in re.findall(r"(\d+)([a-z]*)", s.lower().replace(" ", ""))]
    while strip and k and k[-1] == (0, ""):
        k.pop()
    return tuple(k)


def _has(text: str, kw: str) -> bool:
    """Match a keyword in text. Version-like keywords (containing a digit) match as a
    substring; word keywords match on word boundaries so 'patched' does NOT match
    inside 'unpatched'."""
    kw = kw.lower()
    if any(c.isdigit() for c in kw):
        return kw in text
    return re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text) is not None


def _versions_after(text: str, alias: str) -> set[str]:
    """Versions written right after a product name ('httpd 2.4.49', 'log4j-core 2.14.1',
    'Confluence Server 7.18.0') — at most one qualifier word in between, so the version
    of a neighbouring product is never borrowed."""
    pat = (r"(?<![\w-])" + re.escape(alias.lower()) +
           r"[\w-]*(?:\s+(?!\d)[a-z][\w-]*)?[\s/:v]*(" + _VER + ")")
    return set(re.findall(pat, text))


def _in_range(v: str, r: dict):
    """True/False for version `v` against one range, or None when it cannot decide.
    `affected` marks an end-of-life line that never got a fix."""
    if r.get("affected"):
        return True
    intro, fixed = r.get("introduced"), r.get("fixed_in")
    if not (intro or fixed):
        return None
    k, written = _vkey(v), _vkey(v, strip=False)
    for bound in (intro, fixed):
        bk = _vkey(bound) if bound else ()
        # 'grafana 8.3' names a release line, not a release: vs fix 8.3.1 it can't decide.
        # Three components ('struts 2.5.10' vs 2.5.10.1) name a release and compare.
        if len(written) < min(3, len(bk)) and bk[:len(written)] == written:
            return None
    if intro and k < _vkey(intro):
        return False
    if fixed and k >= _vkey(fixed):
        return False
    return True


# Seed ranges for CVEs the security pack uses. Anything not encoded (or a version that
# cannot be tied to the product) returns None → defer. Keys:
#   products     : names the version must follow; none present → defer
#   requires_all : every keyword must be present (else defer)
#   excludes     : explicit evidence of not-affected → False (checked first)
#   includes     : explicit evidence of affected → True
#   introduced / fixed_in : the default affected range
#   branches     : per-release-line ranges (backported fixes); `prefix` matches leading
#                  version components; a branch with no range defers
#   defer_if     : substrings of the version that mean "can't decide" (vendor hotfixes)
_SEED: dict[str, dict] = {
    "CVE-2023-22515": {"products": ["confluence"], "introduced": "8.0.0", "fixed_in": "8.5.2",
                        "branches": [{"prefix": "8.3", "introduced": "8.0.0", "fixed_in": "8.3.3"},
                                     {"prefix": "8.4", "fixed_in": "8.4.3"}]},
    "CVE-2021-26084": {"products": ["confluence"], "fixed_in": "7.13.0",
                        "branches": [{"prefix": "6.13", "fixed_in": "6.13.23"},
                                     {"prefix": "7.4", "fixed_in": "7.4.11"},
                                     {"prefix": "7.11", "fixed_in": "7.11.6"},
                                     {"prefix": "7.12", "fixed_in": "7.12.5"}]},
    "CVE-2022-42889": {"products": ["commons-text", "commons text"], "introduced": "1.5", "fixed_in": "1.10"},
    "CVE-2022-22965": {"products": ["spring framework", "spring-webmvc", "spring-webflux", "spring-beans"],
                        "fixed_in": "5.3.18", "branches": [{"prefix": "5.2", "fixed_in": "5.2.20"}],
                        "excludes": ["standalone", "executable jar"]},
    "CVE-2021-44228": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.15.0",
                        "branches": [{"prefix": "2.12", "fixed_in": "2.12.2"},
                                     {"prefix": "2.3", "fixed_in": "2.3.1"}]},
    "CVE-2021-43798": {"products": ["grafana"], "introduced": "8.0.0", "fixed_in": "8.3.1",
                        "branches": [{"prefix": "8.0", "fixed_in": "8.0.7"},
                                     {"prefix": "8.1", "fixed_in": "8.1.8"},
                                     {"prefix": "8.2", "fixed_in": "8.2.7"}]},
    "CVE-2022-3602":  {"products": ["openssl"], "introduced": "3.0.0", "fixed_in": "3.0.7"},
    "CVE-2022-0847":  {"products": ["kernel"], "introduced": "5.8", "fixed_in": "5.16.11",
                        "branches": [{"prefix": "5.10", "fixed_in": "5.10.102"},
                                     {"prefix": "5.15", "fixed_in": "5.15.25"}]},
    "CVE-2023-7028":  {"products": ["gitlab"], "introduced": "16.1", "fixed_in": "16.7.2",
                        "branches": [{"prefix": "16.5", "fixed_in": "16.5.6"},
                                     {"prefix": "16.6", "fixed_in": "16.6.4"}]},
    "CVE-2017-5638":  {"products": ["struts"],
                        "branches": [{"prefix": "2.3", "introduced": "2.3.5", "fixed_in": "2.3.32"},
                                     {"prefix": "2.5", "fixed_in": "2.5.10.1"}]},
    "CVE-2024-23897": {"products": ["jenkins"], "fixed_in": "2.442",
                        "branches": [{"prefix": "2.426", "fixed_in": "2.426.3"}]},
    "CVE-2024-4577":  {"products": ["php"], "requires_all": ["windows"], "excludes": ["linux"],
                        "fixed_in": "8.3.8",
                        "branches": [{"prefix": "8.2", "fixed_in": "8.2.20"},
                                     {"prefix": "8.1", "fixed_in": "8.1.29"}]},
    "CVE-2021-40438": {"products": ["httpd", "apache http server", "apache/"], "fixed_in": "2.4.49"},
    "CVE-2023-4966":  {"products": ["netscaler", "citrix adc", "citrix gateway"],
                        "branches": [{"prefix": "14.1", "fixed_in": "14.1-8.50"},
                                     {"prefix": "13.1", "fixed_in": "13.1-49.15"},
                                     {"prefix": "13.0", "fixed_in": "13.0-92.19"},
                                     {"prefix": "12.1", "affected": True}]},
    "CVE-2014-0160":  {"products": ["openssl"], "introduced": "1.0.1", "fixed_in": "1.0.1g"},
    "CVE-2019-0708":  {"excludes": ["2019", "2016", "2012", "windows 10", "windows 11"],
                        "includes": ["windows 7", "server 2008", "server 2003", "windows xp", "vista"]},
    "CVE-2017-0144":  {"excludes": ["disabled", "server 2019", "server 2016", "server 2012"],
                        "includes": ["smbv1 enabled", "unpatched"]},
    "CVE-2021-26855": {"products": ["exchange"], "excludes": ["fully patched", "patched"],
                        "includes": ["without", "unpatched", "pre-2021"]},
    "CVE-2016-5195":  {"products": ["kernel"], "introduced": "2.6.22", "fixed_in": "4.8.3",
                        "branches": [{"prefix": "4.7", "fixed_in": "4.7.9"},
                                     {"prefix": "4.4", "fixed_in": "4.4.26"},
                                     {"prefix": "3"}]},   # 3.x LTS lines got fixes too: defer
    "CVE-2014-6271":  {"products": ["bash"], "excludes": ["windows", "no gnu bash", "no bash"]},
    "CVE-2020-14882": {"products": ["weblogic"]},
    "CVE-2018-7600":  {"products": ["drupal"],
                        "branches": [{"prefix": "8.3", "fixed_in": "8.3.9"},
                                     {"prefix": "8.4", "fixed_in": "8.4.6"},
                                     {"prefix": "8.5", "fixed_in": "8.5.1"},
                                     {"prefix": "7", "fixed_in": "7.58"}]},
    # --- Web servers ---
    "CVE-2020-1938":  {"products": ["tomcat"],
                        "branches": [{"prefix": "9", "fixed_in": "9.0.31"},
                                     {"prefix": "8", "fixed_in": "8.5.51"},
                                     {"prefix": "7", "fixed_in": "7.0.100"}]},
    "CVE-2021-23017": {"products": ["nginx"], "introduced": "0.6.18", "fixed_in": "1.21.0",
                        "branches": [{"prefix": "1.20", "fixed_in": "1.20.1"}]},
    "CVE-2017-7269":  {"products": ["iis"], "introduced": "6.0", "fixed_in": "7.0"},
    "CVE-2021-41773": {"products": ["httpd", "apache http server", "apache/"],
                        "introduced": "2.4.49", "fixed_in": "2.4.50"},
    "CVE-2021-42013": {"products": ["httpd", "apache http server", "apache/"],
                        "introduced": "2.4.49", "fixed_in": "2.4.51"},
    # --- CMS ---
    "CVE-2023-32243": {"products": ["essential addons for elementor", "essential addons"],
                        "introduced": "5.4.0", "fixed_in": "5.7.2"},
    "CVE-2023-23752": {"products": ["joomla"], "introduced": "4.0.0", "fixed_in": "4.2.8"},
    "CVE-2022-24086": {"products": ["magento", "adobe commerce"], "introduced": "2.3.3-p1",
                        "fixed_in": "2.4.3-p2", "branches": [{"prefix": "2.3", "introduced": "2.3.3-p1",
                                                              "fixed_in": "2.3.7-p3"}]},
    "CVE-2019-11043": {"products": ["php-fpm", "php"], "requires_all": ["nginx"],
                        "introduced": "7.1.0", "fixed_in": "7.3.11",
                        "branches": [{"prefix": "7.2", "fixed_in": "7.2.24"},
                                     {"prefix": "7.1", "fixed_in": "7.1.33"}]},
    # --- Java ecosystem ---
    "CVE-2021-45046": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.16.0",
                        "branches": [{"prefix": "2.12", "fixed_in": "2.12.2"},
                                     {"prefix": "2.3"}]},   # 2.3.x status differs by advisory: defer
    "CVE-2021-45105": {"products": ["log4j"], "introduced": "2.0", "fixed_in": "2.17.0",
                        "branches": [{"prefix": "2.12", "fixed_in": "2.12.3"},
                                     {"prefix": "2.3", "fixed_in": "2.3.1"}]},
    "CVE-2022-22963": {"products": ["spring cloud function", "spring-cloud-function"], "fixed_in": "3.2.3",
                        "branches": [{"prefix": "3.1", "fixed_in": "3.1.7"}]},
    "CVE-2017-12149": {"products": ["jboss"], "introduced": "5.0", "fixed_in": "7.0",
                        "excludes": ["eap 7"]},
    # Oracle fixes ship as CPU patches on the same version string: version can't decide.
    "CVE-2020-2555":  {"products": ["weblogic", "coherence"]},
    "CVE-2023-21931": {"products": ["weblogic"]},
    # --- Network appliances ---
    "CVE-2024-21762": {"products": ["fortios", "fortigate"],
                        "branches": [{"prefix": "7.4", "fixed_in": "7.4.3"},
                                     {"prefix": "7.2", "fixed_in": "7.2.7"},
                                     {"prefix": "7.0", "fixed_in": "7.0.14"},
                                     {"prefix": "6.4", "fixed_in": "6.4.15"},
                                     {"prefix": "6.2", "fixed_in": "6.2.16"},
                                     {"prefix": "6.0", "affected": True}]},
    "CVE-2022-40684": {"products": ["fortios", "fortigate"],
                        "branches": [{"prefix": "7.0", "fixed_in": "7.0.7"},
                                     {"prefix": "7.2", "fixed_in": "7.2.2"}]},
    "CVE-2023-20198": {"products": ["ios xe"],
                        "branches": [{"prefix": "17.9", "fixed_in": "17.9.4a"},
                                     {"prefix": "17.6", "fixed_in": "17.6.6a"},
                                     {"prefix": "17.3", "fixed_in": "17.3.8a"},
                                     {"prefix": "16.12", "fixed_in": "16.12.10a"}]},
    "CVE-2024-3400":  {"products": ["pan-os"], "defer_if": ["-h"],   # hotfixes carry backports
                        "branches": [{"prefix": "10.2", "fixed_in": "10.2.9"},
                                     {"prefix": "11.0", "fixed_in": "11.0.4"},
                                     {"prefix": "11.1", "fixed_in": "11.1.2"}]},
    # Ivanti shipped mitigations/patch builds, not a clean fixed version: defer.
    "CVE-2023-46805": {"products": ["ivanti", "pulse"]},
    "CVE-2024-21887": {"products": ["ivanti", "pulse"]},
    # --- Mail ---
    "CVE-2021-34473": {"products": ["exchange"], "excludes": ["fully patched", "cu23", "cu22"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2021-34523": {"products": ["exchange"], "excludes": ["fully patched"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2021-31207": {"products": ["exchange"], "excludes": ["fully patched"],
                        "includes": ["2019", "2016", "2013"]},
    "CVE-2022-27925": {"products": ["zimbra"],
                        "branches": [{"prefix": "8.8", "fixed_in": "8.8.15p31.1"},
                                     {"prefix": "9.0", "fixed_in": "9.0.0p24.1"}]},
    "CVE-2022-41040": {"products": ["exchange"],
                        "includes": ["2019", "2016"], "excludes": ["fully patched"]},
    # --- Container / K8s ---
    "CVE-2024-21626": {"products": ["runc"], "introduced": "1.0.0-rc93", "fixed_in": "1.1.12"},
    "CVE-2022-0185":  {"products": ["kernel"], "introduced": "5.1", "fixed_in": "5.16.2",
                        "branches": [{"prefix": "5.4"}, {"prefix": "5.10"}, {"prefix": "5.15"}]},
    "CVE-2022-0811":  {"products": ["cri-o"], "introduced": "1.19", "fixed_in": "1.23.2",
                        "branches": [{"prefix": "1.19", "fixed_in": "1.19.6"},
                                     {"prefix": "1.20", "fixed_in": "1.20.7"},
                                     {"prefix": "1.21", "fixed_in": "1.21.6"},
                                     {"prefix": "1.22", "fixed_in": "1.22.3"}]},
    # --- Database ---
    "CVE-2023-39417": {"products": ["postgresql", "postgres"], "introduced": "11.0", "fixed_in": "15.4",
                        "branches": [{"prefix": "11", "fixed_in": "11.21"},
                                     {"prefix": "12", "fixed_in": "12.16"},
                                     {"prefix": "13", "fixed_in": "13.12"},
                                     {"prefix": "14", "fixed_in": "14.9"}]},
    "CVE-2023-22809": {"products": ["sudo"], "introduced": "1.8.0", "fixed_in": "1.9.12p2"},
    # --- VPN / Remote access ---
    "CVE-2018-13379": {"products": ["fortios", "fortigate"],
                        "branches": [{"prefix": "6.0", "fixed_in": "6.0.5"},
                                     {"prefix": "5.6", "introduced": "5.6.3", "fixed_in": "5.6.8"},
                                     {"prefix": "5.4", "introduced": "5.4.6", "fixed_in": "5.4.13"}]},
    "CVE-2019-19781": {"products": ["netscaler", "citrix adc", "citrix gateway"],
                        "branches": [{"prefix": "13.0", "fixed_in": "13.0.47.24"},
                                     {"prefix": "12.1", "fixed_in": "12.1.55.18"},
                                     {"prefix": "12.0", "fixed_in": "12.0.63.13"},
                                     {"prefix": "11.1", "fixed_in": "11.1.63.15"},
                                     {"prefix": "10.5", "fixed_in": "10.5.70.12"}]},
    "CVE-2023-27997": {"products": ["fortios", "fortigate"],
                        "branches": [{"prefix": "7.2", "fixed_in": "7.2.5"},
                                     {"prefix": "7.0", "fixed_in": "7.0.12"},
                                     {"prefix": "6.4", "fixed_in": "6.4.13"},
                                     {"prefix": "6.2", "fixed_in": "6.2.14"},
                                     {"prefix": "6.0", "fixed_in": "6.0.17"}]},
    # --- Misc high-profile ---
    "CVE-2023-44487": {"products": ["http2", "http/2", "h2"],
                        "excludes": ["patched", "mitigated"]},
    "CVE-2024-6387":  {"products": ["openssh"], "introduced": "8.5p1", "fixed_in": "9.8p1"},
    "CVE-2023-38408": {"products": ["openssh"], "fixed_in": "9.3p2"},
    "CVE-2022-22947": {"products": ["spring cloud gateway", "spring-cloud-gateway"], "fixed_in": "3.1.1",
                        "branches": [{"prefix": "3.0", "fixed_in": "3.0.7"}]},
}


class CveResolver:
    def __init__(self, ranges: dict | None = None):
        self.ranges = ranges if ranges is not None else _SEED

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "CveResolver":
        """Load `cve_ranges.json` from the model dir, else the built-in seed. A corrupt
        file raises: silently dropping the resolver would hand CVE questions back to
        the model, the exact failure routing exists to prevent."""
        p = Path(model_dir) / "cve_ranges.json"
        if p.exists():
            return cls(json.loads(p.read_text()))
        return cls()  # built-in seed

    def resolve(self, cve: str, text: str):
        """Return True/False if confident, else None (defer)."""
        e = self.ranges.get(cve.upper())
        if not e:
            return None
        t = (text or "").lower()
        if any(_has(t, x) for x in e.get("excludes", [])):
            return False  # explicit not-affected evidence (e.g. 'fully patched')
        prods = e.get("products", [])
        if prods and not any(_has(t, p) for p in prods):
            return None   # product not stated: unknown, not "not affected"
        if not all(_has(t, kw) for kw in e.get("requires_all", [])):
            return None   # a precondition (e.g. Windows CGI) is not stated
        if any(_has(t, i) for i in e.get("includes", [])):
            return True
        if not (e.get("fixed_in") or e.get("introduced") or e.get("branches")):
            return None   # no version rule: product matched but nothing decisive
        vers = set().union(*(_versions_after(t, p) for p in prods)) if prods else set()
        if len(vers) != 1 or _DISTRO_RE.search(t):
            return None   # no version, two versions, or a distro package with backports
        v = vers.pop()
        if any(s in v for s in e.get("defer_if", [])):
            return None
        for br in e.get("branches", []):
            pk = _vkey(br["prefix"], strip=False)
            if _vkey(v, strip=False)[:len(pk)] == pk:
                return _in_range(v, br)
        return _in_range(v, e)
