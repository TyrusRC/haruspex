"""Backbone profile + device selection.

Haruspex is a small, local-first decision model. The default backbone is a
ModernBERT encoder (~150M) — non-autoregressive, CPU/edge-friendly. Any Hugging
Face backbone can be swapped in via env (encoder or decoder), so the model stays
general without shipping a large preset.

  HARUSPEX_BACKBONE=<hf-id>     backbone (default answerdotai/ModernBERT-base)
  HARUSPEX_TYPE=encoder|decoder pooling (default encoder)
  HARUSPEX_LORA=1               LoRA-wrap the backbone for training (default off)
"""

from __future__ import annotations

import os

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"
# 512 keeps memory bounded for long-state decisions; raise via env when needed.
MAX_LEN = int(os.environ.get("HARUSPEX_MAX_LEN", "512"))


def resolve_profile(name: str = "") -> dict:
    """Return {backbone, type, lora} from the environment (small encoder by default)."""
    return {
        "backbone": os.environ.get("HARUSPEX_BACKBONE", DEFAULT_BACKBONE),
        "type": os.environ.get("HARUSPEX_TYPE", "encoder").lower(),
        "lora": os.environ.get("HARUSPEX_LORA", "") not in ("", "0", "false", "no"),
    }


# Back-compat: some callers import BACKBONE directly.
BACKBONE = resolve_profile()["backbone"]


def pick_device(prefer: str = "") -> str:
    """Best available torch device: cuda > mps (Apple) > cpu."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
