"""Per-label logits from the encoder, for one adapter or an ensemble of adapters.

gliner2's `classify_text` returns only the top label and its confidence. Its
ClassificationEngine scores EVERY label through the same classifier head; on real rows
it reproduces classify_text's top-1 exactly (confidence to ~1e-7). Full logits unlock
temperature scaling for every decision type, prediction sets, and ensembles.

Ensemble: several LoRA adapters (e.g. seeds of one recipe) attached to ONE base as named
PEFT adapters; each item is scored under each adapter in turn. Seed ensembles are the
standard fix for confident mistakes (Lakshminarayanan et al. 2017): members that agree
on the easy cases tend to disagree on the ones a single model gets confidently wrong.

NOTE (ceiling): members run sequentially, so latency scales with ensemble size, and
switching the active adapter mutates the shared model, so scoring holds a lock. A
batched multi-adapter forward pass (e.g. S-LoRA style) is the upgrade.
"""

from __future__ import annotations

import math
import re
import threading

# gliner2's engine rejects its own prompt markers inside labels and descriptions (they
# would shift logit-to-label alignment). Labels are answers and cannot be rewritten in
# the data, so they are swapped for safe names here and mapped back after scoring.
# Training goes through classify_text's path, which tolerates them; on real rows the
# swap moves confidence by < 0.015 and never changed a label.
_RESERVED = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]|[()]")


def _safe(text: str) -> str:
    return " ".join(_RESERVED.sub(" ", str(text)).split())


def safe_labels(descs: dict) -> tuple[dict, dict]:
    """{label: description} -> ({safe label: safe description}, {safe label: label})."""
    safe, back = {}, {}
    for i, (name, desc) in enumerate(descs.items()):
        key = _safe(name) or f"option {i + 1}"
        while key in safe:
            key = f"{key} #{i + 1}"
        safe[key], back[key] = (_safe(desc) or None), name
    return safe, back


class Scorer:
    def __init__(self, model, members=None, peft=None, batch_size: int = 8):
        self.model = model
        self.members = list(members) if members else [None]   # None = the one loaded adapter
        self.peft = peft
        self.batch_size = batch_size
        self._clf = None
        self._lock = threading.Lock()

    def logits(self, items) -> list[list[dict | None]]:
        """items: [(labels {name: description}, text)] -> per item, one {label: logit}
        per ensemble member (None when the model gave the label no score)."""
        items = list(items)
        if not items:
            return []
        if not hasattr(self.model, "classifier"):
            return [[self._legacy(d, t)] for d, t in items]
        from gliner2.classification.engine import Classifier, ClassificationConfig
        from gliner2.classification.schema import ClassificationSchema
        mapped = [safe_labels(d) for d, _ in items]
        schemas = [ClassificationSchema().single("decision", m[0]) for m in mapped]
        texts = [t for _, t in items]
        cfg = ClassificationConfig(batch_size=self.batch_size)
        out: list[list] = [[] for _ in items]
        with self._lock:
            if self._clf is None:
                self._clf = Classifier(self.model)
            for m in self.members:
                if m is not None:
                    self.peft.set_adapter(m)
                for i, s in enumerate(self._clf.batch_score(texts, schemas, config=cfg)):
                    back = mapped[i][1]
                    out[i].append({back[k]: v for k, v in s.tasks["decision"].items()})
        return out

    def _legacy(self, descs: dict, text: str) -> dict | None:
        """Models without a classifier head (test doubles, other backends): turn the
        top-1 (label, confidence) into logits with the same softmax. A bare label or a
        zero confidence carries no information, so it is reported as unscored."""
        res = self.model.classify_text(text, {"decision": descs}, threshold=0.0,
                                       include_confidence=True).get("decision")
        if not isinstance(res, dict) or not res.get("confidence") or res.get("label") not in descs:
            return None
        p = min(max(float(res["confidence"]), 1e-12), 1 - 1e-12)
        rest = math.log((1 - p) / max(1, len(descs) - 1))
        return {n: (math.log(p) if n == res["label"] else rest) for n in descs}


def softmax(logits: dict, temperature: float = 1.0) -> dict:
    m = max(logits.values())
    e = {k: math.exp((v - m) / temperature) for k, v in logits.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def ensemble_probs(members: list, temperature: float = 1.0) -> dict | None:
    """Mean of the members' temperature-scaled softmax; None if any member is unscored."""
    if not members or any(m is None for m in members):
        return None
    dists = [softmax(m, temperature) for m in members]
    return {k: sum(d[k] for d in dists) / len(dists) for k in dists[0]}
