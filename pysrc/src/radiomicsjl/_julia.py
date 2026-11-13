"""Lazy start-up of the Julia runtime and of Radiomics.jl."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_lock = threading.Lock()
_bridge = None

CUDA_UUID = "052768ef-5323-5732-b1bb-66c8b64840ba"


def _configure_env() -> None:
    # Radiomics.jl is multi-threaded: use all cores unless the user decided otherwise.
    # With more than one Julia thread, juliacall needs Julia to handle signals.
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")


def get_bridge():
    """Start Julia (installing it and Radiomics.jl on first use) and return the bridge module."""
    global _bridge
    if _bridge is not None:
        return _bridge
    with _lock:
        if _bridge is None:
            _configure_env()
            from juliacall import Main as jl  # first import resolves/installs Julia packages

            bridge_file = (Path(__file__).parent / "_bridge.jl").as_posix()
            jl.seval(f'include("{bridge_file}")')
            _bridge = jl.RadiomicsJLBridge
    return _bridge


def install_gpu(version: str = "6.4.2") -> None:
    """Add CUDA.jl to the Julia environment used by radiomicsjl (needed for use_gpu=True).

    Restart Python afterwards if Julia was already started in this session.
    """
    import juliapkg

    juliapkg.add("CUDA", CUDA_UUID, version=version)
    juliapkg.resolve()
