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
    """Best available torch device: cuda > mps (Apple) > cpu. Always prefer a GPU
    when one is present. With several CUDA GPUs, pick the one with the most free
    memory (the most capable to run on right now)."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n <= 1:
            return "cuda"
        best, best_free = 0, -1
        for i in range(n):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001 - some backends lack per-index query
                return "cuda"
            if free > best_free:
                best, best_free = i, free
        return f"cuda:{best}"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
