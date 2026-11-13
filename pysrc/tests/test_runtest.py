"""The Radiomics.jl test suite (test/runtests.jl), run through the Python wrapper.

Reference values, tolerances, data URLs and DRO settings are read from the Julia
file at test time, so the two suites can never drift apart: changing a value in
runtests.jl changes it here too.

What is run (same as runtests.jl):
  - IBSI digital phantom, 3D            (n_bins=6, keep_largest_only=False)
  - IBSI digital phantom, slices_2d     ((3, 3) in Julia = (2, 2) here, 0-based)
  - single-slice label in a 3D image    (shape3d keys, TotalEnergy, flatness)
  - PET DROs, SUVbw normalisation       (marker "dro": downloads ~60 DICOM series)
  - all of the above on GPU             (marker "gpu": needs an NVIDIA GPU and
                                         `python -m radiomicsjl install-gpu`)

    pytest pysrc/tests/test_ibsi.py -v                 # everything available
    pytest pysrc/tests/test_ibsi.py -m "not dro"       # skip the DRO download

Set RADIOMICSJL_RUNTESTS=/path/to/runtests.jl to use another copy of the Julia tests.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import radiomicsjl as rjl

# ----------------------------------------------------------------------------- runtests.jl
RUNTESTS = Path(os.environ.get("RADIOMICSJL_RUNTESTS",
                               Path(__file__).resolve().parents[2] / "test" / "runtests.jl"))
if not RUNTESTS.is_file():
    pytest.skip(f"Julia test suite not found at {RUNTESTS}", allow_module_level=True)

_SRC = RUNTESTS.read_text(encoding="utf-8")


def _const(name: str) -> str:
    m = re.search(rf'^const {name}\s*=\s*"([^"]*)"', _SRC, re.M)
    assert m, f"const {name} not found in {RUNTESTS}"
    return m.group(1)


PHANTOM_URL = _const("PHANTOM_DIR")
MASK_URL = _const("MASK_DIR")
DRO_REF = _const("DRO_REF")
SUV_NOMINAL_RTOL = float(re.search(r"^const SUV_NOMINAL_RTOL\s*=\s*([0-9.eE+-]+)", _SRC, re.M).group(1))
_err_block = re.search(r"^const DRO_ERROR_MSG = Dict\((.*?)^\)", _SRC, re.M | re.S).group(1)
DRO_ERROR_MSG = dict(re.findall(r'"(DRO_error_[^"]+)"\s*=>\s*"([^"]*)"', _err_block))


def _testset_refs(name: str):
    """(feature, reference, tolerance) of every ibsi_test in the testset `name` (CPU version)."""
    start = _SRC.index(f'@testset "{name}" begin')
    nxt = _SRC.find("@testset", start + 1)
    block = _SRC[start:nxt if nxt != -1 else None]
    refs = re.findall(r'ibsi_test\(\w+\["([^"]+)"\],\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)', block)
    assert refs, f"no ibsi_test found in testset {name!r}"
    return [(k, float(r), float(t)) for k, r, t in refs]


REFS = {"3d": _testset_refs("Radiomics 3D test"), "2d": _testset_refs("Radiomics 2D test")}


def ibsi_test(actual: float, reference: float, tolerance: float) -> bool:
    """Same as ibsi_test in runtests.jl: difference rounded to 3 significant digits of the reference."""
    if reference == 0:
        diff = math.floor(abs(actual - reference))
    else:
        decimals = 3 - (1 + math.floor(math.log10(abs(reference))))
        factor = 10.0 ** decimals
        diff = math.floor(abs(actual - reference) * factor) / factor
    return diff <= tolerance


# ----------------------------------------------------------------------------- GPU
def _nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        return subprocess.run([exe], capture_output=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


def _skip_if_no_gpu(device: str):
    if device == "gpu" and not _nvidia_gpu():
        pytest.skip("no NVIDIA GPU detected (nvidia-smi)")


def _extract(device: str, *args, **kwargs):
    try:
        return rjl.extract_radiomic_features(*args, use_gpu=(device == "gpu"), **kwargs)
    except Exception as e:  # noqa: BLE001
        if device == "gpu" and "CUDA" in str(e) and "could not be loaded" in str(e):
            pytest.skip("CUDA.jl not installed: run `python -m radiomicsjl install-gpu`")
        raise


# ----------------------------------------------------------------------------- data
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


@pytest.fixture(scope="session")
def phantom(tmp_path_factory):
    d = tmp_path_factory.mktemp("ibsi_phantom")
    return str(_download(PHANTOM_URL, d / "phantom.nii.gz")), str(_download(MASK_URL, d / "mask.nii.gz"))


@pytest.fixture(scope="session")
def dro_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dro")
    api = f"https://api.github.com/repos/oncoray/suv_computation/git/trees/{DRO_REF}?recursive=1"
    with urllib.request.urlopen(api, timeout=120) as r:
        tree = json.load(r)
    assert not tree.get("truncated"), "GitHub tree truncated"
    files = [t["path"] for t in tree["tree"]
             if t["type"] == "blob" and t["path"].startswith("DRO/data/") and "/RS/" not in t["path"]]
    raw = f"https://raw.githubusercontent.com/oncoray/suv_computation/{DRO_REF}/"
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: _download(raw + p, d / Path(p).relative_to("DRO/data")), files))
    return d


_cache = {}


def _features(kind: str, device: str, phantom):
    key = (kind, device)
    if key not in _cache:
        img, msk = phantom
        kw = dict(n_bins=6, keep_largest_only=False)  # as in runtests.jl
        if kind == "2d":
            kw["slices_2d"] = [(2, 2)]  # Julia slices_2d=[(3, 3)], 0-based here
        _cache[key] = _extract(device, img, msk, **kw)
    return _cache[key]


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["3d", "2d"])
def test_ibsi_phantom(kind, device, phantom):
    """Every ibsi_test of the 'Radiomics 3D test' / 'Radiomics 2D test' testsets."""
    _skip_if_no_gpu(device)
    f = _features(kind, device, phantom)
    failures = []
    for key, ref, tol in REFS[kind]:
        if key not in f:
            failures.append(f"{key}: missing")
        elif not ibsi_test(f[key], ref, tol):
            failures.append(f"{key}: {f[key]!r} vs reference {ref!r}")
    assert not failures, f"{len(failures)}/{len(REFS[kind])} features fail:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("device", DEVICES)
def test_single_slice_label_in_3d_image(device):
    """Same as the 'Radiomics single-slice label in 3D image' testset."""
    _skip_if_no_gpu(device)
    spacing = [2.0, 3.0, 5.0]
    i, j, k = np.meshgrid(np.arange(1, 21), np.arange(1, 21), np.arange(1, 11), indexing="ij")
    img = (100 + 10 * i + j + 3 * k).astype(np.float32)  # Julia: img[i, j, k], 1-based
    mask = np.zeros((20, 20, 10), np.uint8)
    mask[5:10, 5:10, 4:6] = 1  # Julia mask[6:10, 6:10, 5:6] .= 1   (two slices)
    mask[5:10, 5:10, 7] = 2    # Julia mask[6:10, 6:10, 8]   .= 2   (one slice)
    r = _extract(device, img, mask, spacing, features=["first_order", "shape3d"],
                 labels=[1, 2], keep_largest_only=False)
    for lab in (1, 2):
        v = img[mask == lab].astype(np.float64)
        assert sum(k.startswith("shape3d") for k in r[lab]) == 15
        assert r[lab]["firstorder_total_energy"] == pytest.approx(np.sum(v ** 2) * np.prod(spacing), rel=1.5e-8)
    assert r[2]["shape3d_flatness"] == 0


@pytest.mark.dro
def test_dro_suv(dro_dir):
    """Same as the 'Radiomics DROs set' testset: SUVbw on the QIBA PET DROs."""
    names = sorted(n.name for n in dro_dir.iterdir() if n.is_dir() and n.name.startswith("DRO_"))
    assert names, "no DRO downloaded"
    failures = []
    for n in names:
        pt, msk = str(dro_dir / n / "PT"), str(dro_dir / n / "mask" / "DRO_mask.nii.gz")
        try:
            f = rjl.extract_radiomic_features(pt, msk, suv_ibsi_normalization=True, features=["first_order"])
        except Exception as e:  # noqa: BLE001
            if n.startswith("DRO_error"):
                if DRO_ERROR_MSG[n] not in str(e):
                    failures.append(f"{n}: wrong error: {str(e).splitlines()[0][:120]}")
            else:
                failures.append(f"{n}: unexpected error: {str(e).splitlines()[0][:120]}")
            continue
        if n.startswith("DRO_error"):
            failures.append(f"{n}: expected error {DRO_ERROR_MSG[n]!r}, got features")
            continue
        for key, nominal in (("firstorder_minimum", 0.20), ("firstorder_median", 1.0), ("firstorder_maximum", 4.0)):
            if not math.isclose(f[key], nominal, rel_tol=SUV_NOMINAL_RTOL):
                failures.append(f"{n}: {key} = {f[key]!r}, nominal {nominal}")
    assert not failures, f"{len(failures)} DRO checks fail ({len(names)} DROs):\n  " + "\n  ".join(failures)