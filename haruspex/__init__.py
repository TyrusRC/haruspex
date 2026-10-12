"""Haruspex — a small open-weight decision model.

Reads a situation and returns calibrated typed verdicts (boolean / choice / score),
not text. Per-candidate scoring: score each candidate with a shared `Linear(h, 1)` head,
temperature-scale, and normalize per question. Follows OpenJev's candidate-scoring
recipe; runs on CPU / CUDA / Apple-MPS. A general decision model — train it on any
decision data (a security data pack is provided).
"""

__all__ = ["HaruspexModel"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy so `haruspex.data` / `haruspex.config` import without torch installed.
    if name == "HaruspexModel":
        from .model import HaruspexModel
        return HaruspexModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
