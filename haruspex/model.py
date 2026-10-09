"""HaruspexModel — encoder backbone + per-candidate scorer + calibrated readout.

Design (OpenJev's candidate-scoring recipe, shrunk to an encoder):
  - encode `[STATE][QUESTION][CANDIDATE]` with ModernBERT, mean-pool the last layer
  - a shared `Linear(h, 1)` head -> one scalar per candidate
  - temperature-scale, then normalize per question:
      choice  -> softmax(scores / T)
      boolean -> sigmoid(score / T)   (+ optional isotonic recalibration)
      score   -> softmax(level scores / T), value = Σ i · pᵢ
The 1-D head means the candidate count is free at runtime — no fixed classifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import BACKBONE, MAX_LEN
from .data import normalize


class HaruspexModel(nn.Module):
    def __init__(self, backbone: str = BACKBONE):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        h = self.backbone.config.hidden_size
        self.scorer = nn.Linear(h, 1)
        self.log_temp = nn.Parameter(torch.zeros(()))   # T = exp(log_temp), starts 1.0
        # optional isotonic recalibration of boolean P(yes), as (x, y) for np.interp
        self.iso_x: list[float] = []
        self.iso_y: list[float] = []

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                      # (B, L, H)
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-6)   # mean pool
        return self.scorer(pooled).squeeze(-1)          # (B,) raw score per input

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def score_texts(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """Raw scalar score per text (no temperature)."""
        dev = self._device()
        scores: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt").to(dev)
            scores.append(self.forward(enc["input_ids"], enc["attention_mask"]))
        return torch.cat(scores) if scores else torch.empty(0, device=dev)

    # ---- readout -------------------------------------------------------------

    def _apply_iso(self, p: float) -> float:
        if not self.iso_x:
            return p
        import numpy as np
        return float(np.interp(p, self.iso_x, self.iso_y))

    @torch.no_grad()
    def decide(self, row: dict) -> dict:
        """Answer one on-disk decision row -> the Jev /v1/evaluate answer shape."""
        ex = normalize(row)
        t = float(self.temperature.item())
        s = self.score_texts(ex.texts)
        if ex.type == "boolean":
            p = torch.sigmoid(s[0] / t).item()
            return {"type": "boolean", "probability": self._apply_iso(p)}
        probs = torch.softmax(s / t, dim=0).tolist()
        if ex.type == "choice":
            best = int(max(range(len(probs)), key=lambda i: probs[i]))
            return {"type": "choice", "choice": ex.names[best],
                    "probabilities": {n: p for n, p in zip(ex.names, probs)}}
        # score: expected rubric level
        value = sum(i * p for i, p in enumerate(probs))
        return {"type": "score", "score": value,
                "probabilities": {str(i): p for i, p in enumerate(probs)}}

    # ---- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({"scorer": self.scorer.state_dict(), "log_temp": self.log_temp.detach().cpu()},
                   path / "head.pt")
        (path / "meta.json").write_text(json.dumps(
            {"backbone": self.backbone_name, "iso_x": self.iso_x, "iso_y": self.iso_y}, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "HaruspexModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        m = cls(backbone=str(path))          # loads the fine-tuned backbone + tokenizer from dir
        m.backbone_name = meta.get("backbone", str(path))
        head = torch.load(path / "head.pt", map_location="cpu")
        m.scorer.load_state_dict(head["scorer"])
        with torch.no_grad():
            m.log_temp.copy_(head["log_temp"])
        m.iso_x, m.iso_y = meta.get("iso_x", []), meta.get("iso_y", [])
        return m.to(device).eval()
