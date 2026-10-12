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
    def __init__(self, model, calibrator=None):
        self.model = model
        self.calibrator = calibrator

    @classmethod
    def load(cls, path: str = "", device: str = "cpu", calibrator=None) -> "Haruspex":
        """Load a GLiNER2.5-Decide base, or a Haruspex adapter artifact (a dir with
        `haruspex.json`: the reused base + a LoRA adapter reattached at inference).
        A sibling `calibrator.json` is picked up automatically when present."""
        model_id = resolve_model(path)
        manifest = Path(model_id) / "haruspex.json"
        if manifest.exists():
            from gliner2 import GLiNER2
            spec = json.loads(manifest.read_text())
            model = GLiNER2.from_pretrained(spec["base"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # load_adapter warns as deprecated
                model.load_adapter(str(Path(model_id) / spec.get("adapter", "final")))
        else:
            from gliner2 import AutoExtractor
            model = AutoExtractor.from_pretrained(model_id)
        if device and device != "cpu":
            try:
                model.to(device) if hasattr(model, "to") else model.cuda()
            except Exception:  # noqa: BLE001 - fall back to CPU if the move fails
                pass
        if calibrator is None:
            cal = Path(model_id) / "calibrator.json"
            if cal.exists():
                from .calibrate import SelectiveCalibrator
                calibrator = SelectiveCalibrator.load(cal)
        return cls(model, calibrator)

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
        # format_results without confidence returns the bare label
        return (res or ""), 1.0

    def decide(self, row: dict) -> dict:
        """Return a typed, calibrated verdict for one decision row."""
        t = (row.get("type") or "boolean").lower()
        if t == "noul":
            t = "boolean"
        label, conf = self._classify(row)
        if t == "boolean":
            p_yes = conf if label == "yes" else 1.0 - conf
            verdict = {"type": "boolean", "label": label, "probability": round(p_yes, 6),
                       "confidence": round(conf, 6)}
        elif t == "choice":
            verdict = {"type": "choice", "choice": label, "confidence": round(conf, 6),
                       "probabilities": {label: round(conf, 6)}}
        else:  # score
            levels = [str(x) for x in (row.get("levels") or [])]
            idx = levels.index(label) if label in levels else -1
            verdict = {"type": "score", "score": idx, "level": label,
                       "confidence": round(conf, 6), "probabilities": {label: round(conf, 6)}}
        if self.calibrator is not None:
            verdict.update(self.calibrator.apply(conf))
        return verdict
