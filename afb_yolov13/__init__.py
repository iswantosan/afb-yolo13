"""AFB-YOLOv13 — experiment-side Python utilities.

Modules:
    hypermil: HyperMIL auxiliary head + count loss (v5 novelty).
"""
from .hypermil import (
    AttnMILPool,
    HyperMILHead,
    install_hypermil,
    make_hypermil_callback,
)

__all__ = [
    "AttnMILPool",
    "HyperMILHead",
    "install_hypermil",
    "make_hypermil_callback",
]
