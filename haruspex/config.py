"""Shared config + device selection (CPU / CUDA / Apple-MPS)."""

from __future__ import annotations

import os


# ModernBERT-base: ~150M params, 8192-token context, fast on CPU/MPS/CUDA.
# Swap for "microsoft/deberta-v3-small" by setting HARUSPEX_BACKBONE.
BACKBONE = os.environ.get("HARUSPEX_BACKBONE", "answerdotai/ModernBERT-base")
MAX_LEN = int(os.environ.get("HARUSPEX_MAX_LEN", "1024"))


def pick_device(prefer: str = "") -> str:
    """Return the best available torch device string. cuda > mps (Apple) > cpu."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
