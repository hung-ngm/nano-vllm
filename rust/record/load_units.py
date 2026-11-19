"""Load nanovllm modules without running package ``__init__`` (that imports flash-attn)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "nanovllm"


def install() -> None:
    if "nanovllm" in sys.modules and hasattr(sys.modules["nanovllm"], "__file__"):
        raise RuntimeError("nanovllm was already imported through its __init__")
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [str(PKG)]
    pkg.__package__ = "nanovllm"
    sys.modules["nanovllm"] = pkg
    engine = types.ModuleType("nanovllm.engine")
    engine.__path__ = [str(PKG / "engine")]
    engine.__package__ = "nanovllm.engine"
    sys.modules["nanovllm.engine"] = engine


def exc_text(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"
