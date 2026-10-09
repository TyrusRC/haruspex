"""HaruspexModel — backbone-agnostic per-candidate scorer + calibrated readout.

Encode `[STATE][QUESTION][CANDIDATE]`, pool one vector, score it with a shared
`Linear(h, 1)` head, temperature-scale, normalize per question:
  choice  -> softmax(scores / T)
  boolean -> sigmoid(score / T)   (+ optional isotonic)
  score   -> softmax(level scores / T), value = Σ i · pᵢ

Two backbones share this head:
  - encoder (ModernBERT): mean-pool the last layer. `edge` profile.
  - decoder (Qwen3.5-4B): last real-token hidden; LoRA for efficient training,
    merged into the base at save time so the artifact loads uniformly. `4b` profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .config import MAX_LEN, resolve_profile
from .data import normalize


def _wrap_lora(base):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(base, cfg)


class HaruspexModel(nn.Module):
    def __init__(self, backbone: str = "", backbone_type: str = "", use_lora: bool | None = None):
        super().__init__()
        prof = resolve_profile()
        self.backbone_name = backbone or prof["backbone"]
        self.backbone_type = backbone_type or prof["type"]      # "encoder" | "decoder"
        self.use_lora = prof["lora"] if use_lora is None else use_lora

        base = AutoModel.from_pretrained(self.backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        hidden = base.config.hidden_size
        if self.use_lora:
            base = _wrap_lora(base)
        self.backbone = base
        self.scorer = nn.Linear(hidden, 1)
        self.log_temp = nn.Parameter(torch.zeros(()))   # T = exp(log_temp), starts 1.0
        self.iso_x: list[float] = []
        self.iso_y: list[float] = []

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def _pool(self, hs, attention_mask):
        if self.backbone_type == "decoder":
            idx = attention_mask.sum(1) - 1                 # last real (non-pad) token
            return hs[torch.arange(hs.size(0), device=hs.device), idx]
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)    # encoder: mean-pool
        return (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.scorer(pooled).squeeze(-1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def score_texts(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        dev = self._device()
        scores: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
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
        ex = normalize(row)
        t = float(self.temperature.item())
        s = self.score_texts(ex.texts)
        if ex.type == "boolean":
            return {"type": "boolean", "probability": self._apply_iso(torch.sigmoid(s[0] / t).item())}
        probs = torch.softmax(s / t, dim=0).tolist()
        if ex.type == "choice":
            best = int(max(range(len(probs)), key=lambda i: probs[i]))
            return {"type": "choice", "choice": ex.names[best],
                    "probabilities": dict(zip(ex.names, probs))}
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                "probabilities": {str(i): p for i, p in enumerate(probs)}}

    # ---- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        backbone = self.backbone
        if self.use_lora and hasattr(backbone, "merge_and_unload"):
            backbone = backbone.merge_and_unload()      # fold LoRA -> plain base, uniform load
        backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({"scorer": self.scorer.state_dict(), "log_temp": self.log_temp.detach().cpu()},
                   path / "head.pt")
        (path / "meta.json").write_text(json.dumps(
            {"backbone": self.backbone_name, "type": self.backbone_type,
             "iso_x": self.iso_x, "iso_y": self.iso_y}, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "HaruspexModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        # Load the merged artifact from `path` (never LoRA at inference).
        m = cls(backbone=str(path), backbone_type=meta.get("type", "encoder"), use_lora=False)
        m.backbone_name = meta.get("backbone", str(path))
        head = torch.load(path / "head.pt", map_location="cpu")
        m.scorer.load_state_dict(head["scorer"])
        with torch.no_grad():
            m.log_temp.copy_(head["log_temp"])
        m.iso_x, m.iso_y = meta.get("iso_x", []), meta.get("iso_y", [])
        return m.to(device).eval()
