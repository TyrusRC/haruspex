"""Profiles + device selection.

Two variants, one codebase:
  - edge : ModernBERT encoder (~150M) — runs anywhere (CPU/MPS/8GB), the keyless
           local engine for Praetor/assay. DEFAULT.
  - 4b   : Qwen3.5-4B decoder + LoRA — the JevBench leaderboard contender (the
           eligible small class; Capability = avg(Intelligence, Calibration)).

Select with HARUSPEX_PROFILE=edge|4b; override the base with HARUSPEX_BACKBONE.
"""

from __future__ import annotations

import os

PROFILES: dict[str, dict] = {
    "edge": {"backbone": "answerdotai/ModernBERT-base", "type": "encoder", "lora": False},
    "4b":   {"backbone": "Qwen/Qwen3.5-4B",             "type": "decoder", "lora": True},
}

DEFAULT_PROFILE = os.environ.get("HARUSPEX_PROFILE", "edge")
MAX_LEN = int(os.environ.get("HARUSPEX_MAX_LEN", "1024"))


def resolve_profile(name: str = "") -> dict:
    """Return a profile dict {backbone, type, lora}, honoring env overrides."""
    prof = dict(PROFILES.get(name or DEFAULT_PROFILE, PROFILES["edge"]))
    if os.environ.get("HARUSPEX_BACKBONE"):
        prof["backbone"] = os.environ["HARUSPEX_BACKBONE"]
    return prof


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
