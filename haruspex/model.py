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
from .data import decision_text, label_map, state_text

# Long states are read in overlapping windows: adapters are trained on short states,
# and attention cost grows quadratically. 40 windows covers serve.py's 64K-char cap.
CHUNK_CHARS, CHUNK_OVERLAP, MAX_WINDOWS = 2000, 200, 40


class Haruspex:
    def __init__(self, model, calibrator=None, cve_resolver=None, route_cve=True,
                 cloud_resolver=None, route_cloud=True, novelty=None, scorer=None):
        from .scorer import Scorer
        self.model = model
        # Per-label logits for one adapter or a seed ensemble (scorer.py).
        self.scorer = scorer or Scorer(model)
        self.calibrator = calibrator
        # Optional NoveltyGate (ood.py): abstain on inputs unlike the training data.
        self.novelty = novelty
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
        scorer_args = {}
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
                members = spec.get("adapters") or []
                if len(members) > 1:
                    # seed ensemble: named PEFT adapters on one shared base
                    from peft import PeftModel
                    peft = PeftModel.from_pretrained(model, str(root / members[0]),
                                                     adapter_name=members[0])
                    for m in members[1:]:
                        peft.load_adapter(str(root / m), adapter_name=m)
                    scorer_args = {"members": members, "peft": peft}
                else:
                    # One adapter: merge it into the base. Exact (same dev accuracy) and
                    # ~25% faster on CPU, since no separate LoRA branch runs per layer.
                    from peft import PeftModel
                    adapter = str(root / (members or [spec.get("adapter", "final")])[0])
                    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
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
        from .ood import NoveltyGate
        from .scorer import Scorer
        return cls(model, calibrator, cve_resolver=CveResolver.from_dir(str(root)),
                   cloud_resolver=CloudResolver.from_dir(str(root)),
                   novelty=NoveltyGate.load(root), scorer=Scorer(model, **scorer_args))

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

    def member_logits(self, row: dict) -> list:
        """Raw per-member logits for a row's whole text (temperature fitting)."""
        return self.scorer.logits([(label_map(row), decision_text(row))])[0]

    @staticmethod
    def _windows(row: dict) -> tuple[list[str], bool]:
        """The texts to score for a row, and whether a tail was left unread. A long state
        is split into overlapping windows, each followed by the question."""
        st = state_text(row)
        if len(st) <= CHUNK_CHARS:
            return [decision_text(row, st)], False
        starts = list(range(0, len(st) - CHUNK_OVERLAP, CHUNK_CHARS - CHUNK_OVERLAP))
        return ([decision_text(row, st[i:i + CHUNK_CHARS]) for i in starts[:MAX_WINDOWS]],
                len(starts) > MAX_WINDOWS)

    def _resolve(self, row: dict, t: str) -> dict | None:
        """A deterministic resolver's verdict, or None to ask the model."""
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
        return None

    def decide(self, row: dict) -> dict:
        """Return a typed, calibrated verdict for one decision row."""
        return self.decide_many([row])[0]

    def decide_many(self, rows: list) -> list:
        """Verdicts for many rows. Every model call — across rows and across the windows
        of long states — goes to the encoder as one batch."""
        out: list = [None] * len(rows)
        pending, items = [], []
        for i, row in enumerate(rows):
            t = (row.get("type") or "boolean").lower()
            t = "boolean" if t == "noul" else t
            v = self._resolve(row, t)
            if v is not None:
                out[i] = v
                continue
            descs = label_map(row)
            texts, unread = self._windows(row)
            pending.append((i, t, len(items), len(texts), unread))
            items += [(descs, x) for x in texts]
        scores = self.scorer.logits(items)
        for i, t, start, n, unread in pending:
            out[i] = self._verdict(rows[i], t, scores[start:start + n], unread)
        return out

    def _verdict(self, row: dict, t: str, windows: list, unread: bool) -> dict:
        """Combine per-window, per-member logits into one verdict. Windows: the verdict
        stands only if every window agrees, at the LEAST confident window (the calibrator
        was fitted on single short states, so take no credit for agreement).
        NOTE (ceiling): a decision that needs facts from two distant windows at once is
        not answerable this way; that is a job for an LLM over the whole log."""
        from .scorer import ensemble_probs
        temp = self.calibrator.temperature_for(t) if self.calibrator is not None else 1.0
        dists = [ensemble_probs(m, temp) for m in windows]
        reason, probs, label, conf, pick = None, None, None, 0.0, 0
        if all(d is not None for d in dists):   # else unscored: no label, no confidence
            tops = [max(d, key=d.get) for d in dists]
            pick = min(range(len(dists)), key=lambda j: dists[j][tops[j]])
            if len(set(tops)) > 1:
                pick = max(range(len(dists)), key=lambda j: dists[j][tops[j]])
                reason = f"{len(dists)} windows of a long state disagree"
            elif unread:
                reason = f"state longer than {MAX_WINDOWS} windows; the tail was not read"
            probs, label = dists[pick], tops[pick]
            conf = probs[label]
        full = {k: round(v, 6) for k, v in (probs or {}).items()}
        if t == "boolean":
            verdict = {"type": "boolean", "label": label,
                       "probability": full.get("yes") if probs else None}
        elif t == "choice":
            verdict = {"type": "choice", "choice": label}
        else:  # score
            levels = [str(x) for x in (row.get("levels") or [])]
            verdict = {"type": "score", "score": levels.index(label) if label in levels else -1,
                       "level": label}
        verdict.update({"confidence": round(conf, 6), "probabilities": full, "source": "model"})
        members = windows[pick] if windows else []
        if len(members) > 1 and label is not None:
            from .scorer import softmax
            votes = [max(sm, key=sm.get) for sm in (softmax(m, temp) for m in members)]
            verdict["agreement"] = round(sum(v == label for v in votes) / len(votes), 6)
        if self.calibrator is not None:
            verdict.update(self.calibrator.apply(conf, t))
            if probs:
                ps = self.calibrator.prediction_set(probs, t)
                if ps is not None:
                    verdict["prediction_set"] = ps
        if self.novelty is not None:
            verdict.update(self.novelty.check(decision_text(row)))
            if verdict["novel"]:
                reason = reason or "input is unlike this adapter's training data"
        if reason:
            verdict.update({"accept": False, "abstain": True, "reason": reason})
        if verdict.get("abstain"):
            verdict["route"] = "escalate"   # one signal for callers: a stronger model or a human
        return verdict
