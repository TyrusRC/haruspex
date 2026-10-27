"""Haruspex — a thin, calibrated decision wrapper over GLiNER2.5-Decide.

Load the published encoder (or a fine-tuned copy), classify a decision row into
one typed verdict, and (optionally) gate it through a selective calibrator that
abstains when the model is not confident enough for the operator's risk budget.

The model is REUSED, not rebuilt. Haruspex adds: the typed decision contract
(boolean / choice / score), the specialization data + training wrapper, and the
selective-calibration / abstention layer.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from .config import resolve_model
from .data import decision_text, label_map


class Haruspex:
    def __init__(self, model, calibrator=None, cve_resolver=None, route_cve=True,
                 cloud_resolver=None, route_cloud=True):
        self.model = model
        self.calibrator = calibrator
        # CVE applicability is a knowledge lookup, not a model judgment — route it.
        self.cve_resolver = cve_resolver
        self.route_cve = route_cve
        # Cloud/IaC scanner findings are a structural fact the scanner already computed;
        # a 0.5B encoder can't reliably negate polarity on them — route deterministically.
        self.cloud_resolver = cloud_resolver
        self.route_cloud = route_cloud

    @classmethod
    def load(cls, path: str = "", device: str = "", calibrator=None,
             subfolder: str = "") -> "Haruspex":
        """Load a GLiNER2.5-Decide base, or a Haruspex adapter artifact (a dir with
        `haruspex.json`: the reused base + a LoRA adapter reattached at inference).
        `path` is a local dir or an HF repo id; `subfolder` selects one adapter inside
        a multi-domain repo (e.g. load("TyrusRC/haruspex", subfolder="mobile")). A
        sibling `calibrator.json` and CVE data are picked up from the same dir.
        `device` defaults to the best available (GPU first); pass "cpu" to force CPU."""
        if not device:
            from .config import pick_device
            device = pick_device()
        model_id = resolve_model(path)
        root, passthrough = cls._resolve_dir(model_id, subfolder)
        if passthrough:
            # a plain base model id (no adapter manifest) — gliner2 handles the download
            from gliner2 import AutoExtractor
            model = AutoExtractor.from_pretrained(model_id)
            root = Path(model_id)   # for a local base dir; harmless otherwise
        else:
            manifest = root / "haruspex.json"
            if manifest.exists():
                from gliner2 import GLiNER2
                spec = json.loads(manifest.read_text())
                model = GLiNER2.from_pretrained(spec["base"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")   # load_adapter warns as deprecated
                    model.load_adapter(str(root / spec.get("adapter", "final")))
            else:
                from gliner2 import AutoExtractor
                model = AutoExtractor.from_pretrained(str(root))
        if device and device != "cpu":
            try:
                model.to(device) if hasattr(model, "to") else model.cuda()
            except Exception as e:  # noqa: BLE001 - fall back to CPU, but say so
                warnings.warn(f"haruspex: could not move model to {device} ({e}); running on CPU")
        if hasattr(model, "eval"):
            # Inference must run in eval mode: a model left in train mode keeps
            # dropout active, which was found live to perturb borderline scores
            # run-to-run on the SAME input — real non-determinism, concentrated
            # exactly in the ambiguous/near-boundary cases that matter most.
            model.eval()
        if calibrator is None:
            cal = root / "calibrator.json"
            if cal.exists():
                from .calibrate import SelectiveCalibrator
                calibrator = SelectiveCalibrator.load(cal)
        from .cve import CveResolver
        from .cloud import CloudResolver
        return cls(model, calibrator, cve_resolver=CveResolver.from_dir(str(root)),
                   cloud_resolver=CloudResolver.from_dir(str(root)))

    @staticmethod
    def _resolve_dir(model_id: str, subfolder: str = "") -> tuple[Path, bool]:
        """Return (local_dir_holding_the_adapter, is_base_passthrough).

        A local dir is used as-is. An HF repo id is probed for the adapter manifest:
        present -> snapshot the repo to a local cache and read from there (this is what
        makes `Haruspex.load("<org>/haruspex")` actually reattach the adapter, not just
        load the base); absent -> it is a plain base model id, pass it through to
        gliner2's own downloader. NOTE: snapshot pulls the whole repo (sibling adapter
        subfolders included) — a few extra MB, acceptable; narrow with allow_patterns
        if a repo ever grows large."""
        p = Path(model_id)
        if p.is_dir():
            return (p / subfolder if subfolder else p), False
        rel = f"{subfolder}/haruspex.json" if subfolder else "haruspex.json"
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            hf_hub_download(model_id, rel)   # cheap probe: 404 if not an adapter repo
        except Exception:  # noqa: BLE001 - not an adapter repo -> base passthrough
            return p, True
        snap = Path(snapshot_download(model_id))
        return (snap / subfolder if subfolder else snap), False

    def _classify(self, row: dict) -> tuple[str, float]:
        """Top label + its confidence for one decision row.
        Uses the lightweight dict-form task: GLiNER2's schema path for top_k /
        multi_label is bugged (processor IndexError), so full per-label
        distributions are not available yet — see calibrate.py.
        """
        descs = label_map(row)
        out = self.model.classify_text(decision_text(row), {"decision": descs},
                                       threshold=0.0, include_confidence=True)
        res = out.get("decision")
        if isinstance(res, dict):
            return res.get("label", ""), float(res.get("confidence", 0.0))
        # a bare label carries no confidence: report none rather than a free pass
        return (res or ""), 0.0

    def decide(self, row: dict) -> dict:
        """Return a typed, calibrated verdict for one decision row."""
        t = (row.get("type") or "boolean").lower()
        if t == "noul":
            t = "boolean"
        # Route CVE-applicability to the deterministic resolver (defer if unsure).
        if self.route_cve and t == "boolean" and self.cve_resolver is not None:
            from .cve import extract_cve, is_applicability
            if is_applicability(row):
                cve = extract_cve(f"{row.get('question','')} {row.get('state','')}")
                ans = self.cve_resolver.resolve(cve, str(row.get("state", "")))
                if ans is not None:
                    return {"type": "boolean", "label": "yes" if ans else "no",
                            "probability": 1.0 if ans else 0.0, "confidence": 1.0,
                            "source": "cve_lookup", "cve": cve,
                            "accept": True, "abstain": False}
                return {"type": "boolean", "label": None, "probability": None,
                        "confidence": 0.0, "source": "model_declined", "cve": cve,
                        "route": "external_cve_lookup", "accept": False, "abstain": True,
                        "reason": "CVE applicability is resolved by a KEV/NVD lookup, not the model"}
        # Route cloud/IaC scanner findings to the deterministic resolver (defer if unsure).
        if self.route_cloud and t == "boolean" and self.cloud_resolver is not None:
            from .cloud import is_cloud_finding
            if is_cloud_finding(row):
                ans = self.cloud_resolver.classify(str(row.get("state", "")))
                if ans is not None:
                    return {"type": "boolean", "label": "yes" if ans else "no",
                            "probability": 1.0 if ans else 0.0, "confidence": 1.0,
                            "source": "cloud_rule", "accept": True, "abstain": False}
                # unsure -> fall through to the model
        label, conf = self._classify(row)
        if t == "boolean":
            p_yes = None   # unscored label: no probability to report
            if conf > 0.0:
                p_yes = conf if label == "yes" else 1.0 - conf
                if self.calibrator is not None:
                    p_yes = self.calibrator.scale(p_yes)   # temperature: monotone, label kept
                    conf = max(p_yes, 1.0 - p_yes)
                p_yes = round(p_yes, 6)
            verdict = {"type": "boolean", "label": label, "probability": p_yes,
                       "confidence": round(conf, 6)}
        elif t == "choice":
            verdict = {"type": "choice", "choice": label, "confidence": round(conf, 6),
                       "probabilities": {label: round(conf, 6)}}
        else:  # score
            levels = [str(x) for x in (row.get("levels") or [])]
            idx = levels.index(label) if label in levels else -1
            verdict = {"type": "score", "score": idx, "level": label,
                       "confidence": round(conf, 6), "probabilities": {label: round(conf, 6)}}
        verdict["source"] = "model"
        if self.calibrator is not None:
            verdict.update(self.calibrator.apply(conf, t))
        return verdict
