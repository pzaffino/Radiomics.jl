"""Image filters (filter_type / filter_params) through the Python wrapper.

Every case is run twice: through `rjl.extract_radiomic_features` with Python arguments,
and directly in Julia with the equivalent Julia arguments. The two results must be identical,
so the Python -> Julia conversion of the filter arguments and of the nested result is checked.

    pytest pysrc/tests/test_filters.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import radiomicsjl as rjl

SPACING = [1.0, 1.0, 2.0]
SETTINGS = dict(n_bins=16, keep_largest_only=False)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    img = rng.uniform(0, 100, size=(18, 18, 18))
    img[6:12, 6:12, :] += 50                       # some structure for the filters
    mask = np.zeros(img.shape, np.uint8)
    mask[4:14, 4:14, 4:14] = 1
    return img, mask


@pytest.fixture(scope="module")
def jl():
    rjl.julia_version()                            # start Julia and load the bridge
    from juliacall import Main
    return Main


def julia_reference(jl, img, mask, filter_type, params_jl, features_jl=None, slices_jl=None):
    """Same extraction called directly in Julia, converted with the bridge's own topy."""
    b = jl.RadiomicsJLBridge
    kw = dict(filter_type=filter_type, filter_params=jl.seval(params_jl), **SETTINGS)
    if features_jl is not None:
        kw["features"] = jl.seval(features_jl)
    if slices_jl is not None:
        kw["slices_2d"] = jl.seval(slices_jl)
    res = b.Radiomics.extract_radiomic_features(b._arr(img), b._arr(mask), b._spacing(np.array(SPACING)), **kw)
    return b.topy(res)


def assert_same(a, b, path="result"):
    """Same nested keys and identical values (diagnosis_* excluded: timings)."""
    assert isinstance(a, dict) and isinstance(b, dict), path
    ka = {k for k in a if not (isinstance(k, str) and k.startswith("diagnosis_"))}
    kb = {k for k in b if not (isinstance(k, str) and k.startswith("diagnosis_"))}
    assert ka == kb, f"{path}: different keys {sorted(map(str, ka ^ kb))}"
    for k in ka:
        x, y = a[k], b[k]
        if isinstance(x, dict):
            assert_same(x, y, f"{path}[{k!r}]")
        elif isinstance(x, float):
            assert (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12), \
                f"{path}[{k!r}]: {x!r} vs {y!r}"
        else:
            assert x == y, f"{path}[{k!r}]: {x!r} vs {y!r}"


def feature_dicts(res):
    """All leaf feature dictionaries of a (possibly nested) result."""
    if res and all(isinstance(v, dict) for v in res.values()):
        return [d for v in res.values() for d in feature_dicts(v)]
    return [res]


# ----------------------------------------------------------------------------- Haar wavelet
def test_haar_default_subbands(data, jl):
    img, mask = data
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar", **SETTINGS)
    assert set(f) == {"haar_wavelet-LLL", "haar_wavelet-LLH", "haar_wavelet-LHH", "haar_wavelet-HHH"}
    assert_same(f, julia_reference(jl, img, mask, "haar", "(;)"))


def test_haar_level_and_subbands(data, jl):
    img, mask = data
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar",
                                      filter_params={"level": 2, "subbands": ["LLL", "HHH"]},
                                      features=["first_order", "glcm"], **SETTINGS)
    assert set(f) == {"haar_wavelet-HHH", "haar_wavelet2-HHH", "haar_wavelet2-LLL"}
    ref = julia_reference(jl, img, mask, "haar", '(level=2, subbands=["LLL", "HHH"])', "[:first_order, :glcm]")
    assert_same(f, ref)


def test_no_shape_features_on_filtered_images(data):
    img, mask = data
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar", **SETTINGS)
    for d in feature_dicts(f):
        assert not any(k.startswith(("shape2d_", "shape3d_")) for k in d)
        assert any(k.startswith("glcm_") for k in d)


# ----------------------------------------------------------------------------- LoG and Gabor
def test_log_several_sigmas(data, jl):
    img, mask = data
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="log",
                                      filter_params={"sigma": [1.0, 2.0]}, **SETTINGS)
    assert len(f) == 2 and all(k.startswith("log_log-sigma-") for k in f)
    assert_same(f, julia_reference(jl, img, mask, "log", "(sigma=[1.0, 2.0],)"))


def test_gabor_single_map(data, jl):
    img, mask = data
    params = {"sigma": 3.0, "lambda": 4.0, "gamma": 1.5, "rotation_invariant": True, "delta_theta": math.pi / 4}
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="gabor", filter_params=params,
                                      features=["first_order"], **SETTINGS)
    assert not any(isinstance(v, dict) for v in f.values())      # one response map: flat result
    ref = julia_reference(jl, img, mask, "gabor",
                          f"(sigma=3.0, lambda=4.0, gamma=1.5, rotation_invariant=true, delta_theta={math.pi / 4!r})",
                          "[:first_order]")
    assert_same(f, ref)


# ----------------------------------------------------------------------------- with slices_2d
def test_filter_with_slices_2d_is_0_based(data, jl):
    img, mask = data
    f = rjl.extract_radiomic_features(img, mask, SPACING, filter_type="log", filter_params={"sigma": [1.0, 2.0]},
                                      features=["first_order"], slices_2d=[(2, 5), (2, 6)], **SETTINGS)
    for per_map in f.values():
        assert set(per_map) == {(2, 5), (2, 6)}
    ref = julia_reference(jl, img, mask, "log", "(sigma=[1.0, 2.0],)", "[:first_order]", "[(3, 6), (3, 7)]")
    assert_same(f, rjl._slices_to_0_based(ref))


# ----------------------------------------------------------------------------- invalid arguments
def test_invalid_filter_arguments(data):
    img, mask = data
    with pytest.raises(Exception, match="(?i)filter"):
        rjl.extract_radiomic_features(img, mask, SPACING, filter_type="foo", **SETTINGS)
    with pytest.raises(TypeError):
        rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar", filter_params=[("level", 2)])
    with pytest.raises(TypeError):
        rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar", filter_params={"level": {"a": 1}})
    with pytest.raises(ValueError):
        rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar", filter_params={"not valid": 1})
    with pytest.raises(ValueError):
        rjl.extract_radiomic_features(img, mask, SPACING, filter_type="haar,log")
