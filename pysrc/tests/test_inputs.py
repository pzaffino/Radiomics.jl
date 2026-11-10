"""Input routes of the Python wrapper, on the IBSI digital phantom (real NIfTI files).

The phantom and its mask are downloaded from the URLs in test/runtests.jl, read with
SimpleITK and passed to the wrapper in every supported way. All routes must give the
same features as the file paths read by Radiomics.jl, and the SimpleITK route must
also pass the IBSI reference values of test/runtests.jl.

    pytest pysrc/tests/test_inputs.py -v
"""

from __future__ import annotations

import math
import os
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import radiomicsjl as rj

sitk = pytest.importorskip("SimpleITK")

RUNTESTS = Path(os.environ.get("RADIOMICSJL_RUNTESTS",
                               Path(__file__).resolve().parents[2] / "test" / "runtests.jl"))
if not RUNTESTS.is_file():
    pytest.skip(f"Julia test suite not found at {RUNTESTS}", allow_module_level=True)
_SRC = RUNTESTS.read_text(encoding="utf-8")

# Axis-named 2D diameters swap names when the array axes are permuted (numpy (z, y, x) route)
DIAM_2D = ("shape3d_maximum_2d_diameter_slice", "shape3d_maximum_2d_diameter_row",
           "shape3d_maximum_2d_diameter_column")
SETTINGS = dict(n_bins=6, keep_largest_only=False)  # as in runtests.jl


def _const(name: str) -> str:
    return re.search(rf'^const {name}\s*=\s*"([^"]*)"', _SRC, re.M).group(1)


def _refs(testset: str):
    start = _SRC.index(f'@testset "{testset}" begin')
    nxt = _SRC.find("@testset", start + 1)
    block = _SRC[start:nxt if nxt != -1 else None]
    return [(k, float(r), float(t)) for k, r, t in
            re.findall(r'ibsi_test\(\w+\["([^"]+)"\],\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)', block)]


def ibsi_test(actual: float, reference: float, tolerance: float) -> bool:
    if reference == 0:
        diff = math.floor(abs(actual - reference))
    else:
        factor = 10.0 ** (3 - (1 + math.floor(math.log10(abs(reference)))))
        diff = math.floor(abs(actual - reference) * factor) / factor
    return diff <= tolerance


def assert_same(a: dict, b: dict, skip=()):
    """Same keys and identical values (diagnosis_* excluded: timings, sizes)."""
    ka = {k for k in a if not k.startswith("diagnosis_") and k not in skip}
    kb = {k for k in b if not k.startswith("diagnosis_") and k not in skip}
    assert ka == kb, f"different keys: {sorted(ka ^ kb)}"
    diff = [k for k in sorted(ka)
            if not (a[k] == b[k] or (isinstance(a[k], float) and math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12))
                    or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])))]
    assert not diff, "different values:\n  " + "\n  ".join(f"{k}: {a[k]!r} vs {b[k]!r}" for k in diff)


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    d = tmp_path_factory.mktemp("ibsi_phantom_inputs")
    paths = []
    for name, url in (("phantom.nii.gz", _const("PHANTOM_DIR")), ("mask.nii.gz", _const("MASK_DIR"))):
        dest = d / name
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        paths.append(str(dest))
    img, msk = paths
    return {"img": img, "msk": msk, "s_img": sitk.ReadImage(img), "s_msk": sitk.ReadImage(msk),
            "ref": rj.extract_radiomic_features(img, msk, **SETTINGS)}


# ----------------------------------------------------------------------------- SimpleITK
def test_simpleitk_equals_path(phantom):
    f = rj.extract_radiomic_features(phantom["s_img"], phantom["s_msk"], **SETTINGS)
    assert_same(phantom["ref"], f)


def test_simpleitk_passes_ibsi_references(phantom):
    f = rj.extract_radiomic_features(phantom["s_img"], phantom["s_msk"], **SETTINGS)
    refs = _refs("Radiomics 3D test")
    fail = [f"{k}: {f.get(k)!r} vs {r!r}" for k, r, t in refs if k not in f or not ibsi_test(f[k], r, t)]
    assert not fail, f"{len(fail)}/{len(refs)} fail:\n  " + "\n  ".join(fail)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slices_2d_simpleitk_equals_path(phantom, axis):
    m = sitk.GetArrayViewFromImage(phantom["s_msk"]).transpose()  # (x, y, z), as the file readers
    k = int(np.argmax((m > 0).sum(axis=tuple(a for a in range(3) if a != axis))))
    a = rj.extract_radiomic_features(phantom["img"], phantom["msk"], slices_2d=[(axis, k)], **SETTINGS)
    b = rj.extract_radiomic_features(phantom["s_img"], phantom["s_msk"], slices_2d=[(axis, k)], **SETTINGS)
    assert_same(a, b)


# ----------------------------------------------------------------------------- numpy
def test_numpy_zyx_with_reversed_spacing_equals_path(phantom):
    arr_i = sitk.GetArrayFromImage(phantom["s_img"])  # (z, y, x)
    arr_m = sitk.GetArrayFromImage(phantom["s_msk"])
    f = rj.extract_radiomic_features(arr_i, arr_m, phantom["s_img"].GetSpacing()[::-1], **SETTINGS)
    assert_same(phantom["ref"], f, skip=DIAM_2D)


def test_numpy_xyz_equals_path(phantom):
    arr_i = sitk.GetArrayFromImage(phantom["s_img"]).transpose()  # (x, y, z)
    arr_m = sitk.GetArrayFromImage(phantom["s_msk"]).transpose()
    f = rj.extract_radiomic_features(arr_i, arr_m, phantom["s_img"].GetSpacing(), **SETTINGS)
    assert_same(phantom["ref"], f)


def test_numpy_2d_slice_equals_slices_2d(phantom):
    """A 2D numpy slice gives the same first-order features as slices_2d on that slice (0-based)."""
    arr_i = sitk.GetArrayFromImage(phantom["s_img"]).transpose()
    arr_m = sitk.GetArrayFromImage(phantom["s_msk"]).transpose()
    k = int(np.argmax((arr_m > 0).sum(axis=(0, 1))))
    a = rj.extract_radiomic_features(phantom["img"], phantom["msk"], slices_2d=[(2, k)],
                                     features=["first_order"], **SETTINGS)
    b = rj.extract_radiomic_features(np.ascontiguousarray(arr_i[:, :, k]), np.ascontiguousarray(arr_m[:, :, k]),
                                     phantom["s_img"].GetSpacing()[:2], features=["first_order"], **SETTINGS)
    for key in ("firstorder_mean", "firstorder_minimum", "firstorder_maximum", "firstorder_median"):
        assert a[key] == pytest.approx(b[key], rel=1e-12), key


# ----------------------------------------------------------------------------- options
def test_bin_width_and_feature_selection(phantom):
    f = rj.extract_radiomic_features(phantom["img"], phantom["msk"], bin_width=1,
                                     features=["first_order", "glcm"], keep_largest_only=False)
    assert any(k.startswith("glcm_") for k in f)
    assert any(k.startswith("firstorder_") for k in f)
    assert not any(k.startswith(("shape3d_", "glrlm_", "glszm_", "gldm_", "ngtdm_")) for k in f)


# ----------------------------------------------------------------------------- invalid inputs
def test_invalid_inputs(phantom):
    arr = sitk.GetArrayFromImage(phantom["s_img"])
    with pytest.raises(TypeError):                     # path mixed with array
        rj.extract_radiomic_features(phantom["img"], arr, [1, 1, 1])
    with pytest.raises(TypeError):                     # SimpleITK mixed with array
        rj.extract_radiomic_features(phantom["s_img"], arr, [1, 1, 1])
    with pytest.raises(ValueError):                    # 4D array
        rj.extract_radiomic_features(arr[None], arr[None], [1, 1, 1])
    with pytest.raises(ValueError):                    # slices_2d axis out of 0..2
        rj.extract_radiomic_features(phantom["img"], phantom["msk"], slices_2d=[(3, 0)])
    with pytest.raises(ValueError):                    # negative slice index
        rj.extract_radiomic_features(phantom["img"], phantom["msk"], slices_2d=[(2, -1)])
    other = sitk.Image(phantom["s_msk"])               # mask with a different spacing
    other.SetSpacing([s * 2 for s in phantom["s_msk"].GetSpacing()])
    with pytest.raises(ValueError):
        rj.extract_radiomic_features(phantom["s_img"], other)