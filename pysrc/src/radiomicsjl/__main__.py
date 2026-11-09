"""Command line helpers.

    python -m radiomicsjl setup        # install Julia + Radiomics.jl now (otherwise done at first use)
    python -m radiomicsjl install-gpu  # add CUDA.jl for use_gpu=True
    python -m radiomicsjl version
"""

import argparse

from . import __version__, install_gpu, julia_version


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m radiomicsjl")
    ap.add_argument("command", choices=["setup", "install-gpu", "version"])
    args = ap.parse_args()

    if args.command == "install-gpu":
        install_gpu()
        print("CUDA.jl installed. Restart Python, then use use_gpu=True.")
    else:
        jl = julia_version()  # starts Julia, installing everything if needed
        print(f"radiomicsjl {__version__} · Radiomics.jl {jl}")


if __name__ == "__main__":
    main()
