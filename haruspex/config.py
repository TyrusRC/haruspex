"""Model selection + device.

Haruspex reuses a published decision encoder — **GLiNER2.5-Decide** (fastino,
Apache-2.0, ~494M params, DeBERTa-family, non-autoregressive). Haruspex does NOT
train a decision model from scratch; it specializes this base on your decisions
and wraps it with a selective-calibration layer.

  HARUSPEX_MODEL=<hf-id-or-dir>   base or fine-tuned model (default below)
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "fastino/GLiNER2.5-Decide"


def resolve_model(path: str = "") -> str:
    """The model id/dir to load: explicit arg > env > default base."""
    return path or os.environ.get("HARUSPEX_MODEL", "") or DEFAULT_MODEL


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
