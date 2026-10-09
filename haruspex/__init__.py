"""Haruspex — a small open-weight decision model.

Reads a situation and returns calibrated typed verdicts (boolean / choice / score),
not text. Per-candidate scoring on a ModernBERT encoder: score each candidate with a
shared `Linear(h, 1)` head, temperature-scale, and normalize per question. The design
follows OpenJev's candidate-scoring recipe on a tiny encoder backbone so it runs on
CPU / CUDA / Apple-MPS from 8 GB RAM, and specializes on in-domain (security) decisions
to beat a big general decoder on the axes that matter: domain accuracy, calibration
(ECE/Brier), latency, and footprint.
"""

__all__ = ["HaruspexModel"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy so `haruspex.data` / `haruspex.config` import without torch installed.
    if name == "HaruspexModel":
        from .model import HaruspexModel
        return HaruspexModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
