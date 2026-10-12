"""Haruspex — a small, local, calibrated decision layer for security operations.

Reuses the published GLiNER2.5-Decide encoder (not a from-scratch model), adds a
typed decision contract (boolean / choice / score), a specialization-training
wrapper, and a selective-calibration / abstention layer. Reads a situation,
returns a calibrated typed verdict — and abstains when it should.
"""

__all__ = ["Haruspex"]
__version__ = "0.2.0"


def __getattr__(name):
    # Lazy so `haruspex.data` / `haruspex.config` import without torch/gliner2.
    if name == "Haruspex":
        from .model import Haruspex
        return Haruspex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
