"""Start-up of the Julia runtime and of Radiomics.jl (called when radiomicsjl is imported)."""

from __future__ import annotations

import os
import threading
import sys
from pathlib import Path

_lock = threading.Lock()
_bridge = None

CUDA_UUID = "052768ef-5323-5732-b1bb-66c8b64840ba"


def _needs_setup() -> bool:
    """No Manifest.toml in the juliapkg project: Julia packages not installed yet."""
    try:
        # juliapkg.project() would resolve (install) first: read the path without resolving
        from juliapkg.state import STATE

        project = STATE["project"]
    except Exception:
        project = os.environ.get("PYTHON_JULIAPKG_PROJECT") or os.path.join(sys.prefix, "julia_env")
    return not os.path.isfile(os.path.join(project, "Manifest.toml"))


def _configure_env() -> None:
    # Radiomics.jl is multi-threaded: use all cores unless the user decided otherwise.
    # With more than one Julia thread, juliacall needs Julia to handle signals.
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")


def _fast_exit_in_workers() -> None:
    """In multiprocessing workers, skip Python/Julia finalization at exit: it is slow,
    and Pool.terminate() would SIGTERM the worker meanwhile, making Julia print a
    backtrace of every thread."""
    import atexit

    def _exit_now():
        import multiprocessing

        # checked at exit time: spawn workers import radiomicsjl before
        # multiprocessing knows they are workers
        if multiprocessing.parent_process() is None:
            return
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)

    # registered after juliacall's own atexit hook, so it runs first
    atexit.register(_exit_now)


def get_bridge():
    """Start Julia (installing it and Radiomics.jl on first use) and return the bridge module."""
    global _bridge
    if _bridge is not None:
        return _bridge
    with _lock:
        if _bridge is None:
            _configure_env()

            first_time = _needs_setup()
            if first_time:
                print("radiomicsjl: first-time setup, installing Julia and Radiomics.jl,this takes a few minutes and happens only once,for this Python environment.", file=sys.stderr, flush=True)

            from juliacall import Main as jl  # first import resolves/installs Julia packages

            if first_time:
                print("radiomicsjl: setup complete.", file=sys.stderr, flush=True)

            _fast_exit_in_workers()

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