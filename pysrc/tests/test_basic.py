"""End-to-end tests: they start Julia (and install it the first time)."""

import numpy as np
import pytest

import radiomicsjl as rj


@pytest.fixture(scope="module")
def cube():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, size=(10, 10, 10))
    msk = np.zeros((10, 10, 10), dtype=np.uint8)
    msk[2:8, 2:8, 2:8] = 1
    return img, msk


def test_version():
    assert rj.julia_version().count(".") == 2


def test_3d_array(cube):
    img, msk = cube
    f = rj.extract_radiomic_features(img, msk, [1.0, 1.0, 1.0], n_bins=16)
    assert isinstance(f, dict)
    assert f["shape3d_voxel_volume"] == pytest.approx(216.0)
    assert f["firstorder_mean"] == pytest.approx(img[msk == 1].mean())
    assert all(isinstance(k, str) for k in f)


def test_axis_order_is_numpy_order(cube):
    img, msk = cube
    m = np.zeros_like(msk)
    m[2:8, 2:8, 4] = 1  # single slice along the last axis
    f = rj.extract_radiomic_features(img, m, [1.0, 1.0, 3.0], n_bins=16)
    assert f["shape3d_voxel_volume"] == pytest.approx(36 * 3.0)
    assert f["firstorder_mean"] == pytest.approx(img[m == 1].mean())


def test_bin_width_and_features(cube):
    img, msk = cube
    f = rj.extract_radiomic_features(img, msk, (1, 1, 1), bin_width=25, features=["glcm", "first_order"])
    assert any(k.startswith("glcm_") for k in f)
    assert not any(k.startswith("shape3d_") for k in f)


def test_multi_label(cube):
    img, msk = cube
    m = msk.astype(np.int64)
    m[2:8, 2:8, 5:8] = 2
    f = rj.extract_radiomic_features(img, m, [1, 1, 1], labels=[1, 2], n_bins=16)
    assert set(f) == {1, 2}


def test_slices_2d(cube):
    img, msk = cube
    f = rj.extract_radiomic_features(img, msk, [1, 1, 1], n_bins=16, slices_2d=[(2, 3), (2, 4)])
    assert set(f) == {(2, 3), (2, 4)}


def test_slices_2d_is_0_based(cube):
    img, msk = cube
    f = rj.extract_radiomic_features(img, msk, [1, 1, 1], n_bins=16, slices_2d=[(2, 4)])
    assert f["firstorder_mean"] == pytest.approx(img[:, :, 4][msk[:, :, 4] == 1].mean())


def test_2d_array():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 100, size=(12, 12))
    msk = np.zeros((12, 12), dtype=bool)
    msk[3:9, 3:9] = True
    f = rj.extract_radiomic_features(img, msk, [1.0, 1.0], n_bins=8)
    assert any(k.startswith("shape2d_") for k in f)


def test_simpleitk_matches_xyz_array(cube):
    sitk = pytest.importorskip("SimpleITK")
    img, msk = cube  # (x, y, z)
    spacing = (0.8, 0.9, 3.0)
    s_img = sitk.GetImageFromArray(np.ascontiguousarray(img.transpose()))  # sitk arrays are (z, y, x)
    s_msk = sitk.GetImageFromArray(np.ascontiguousarray(msk.transpose()))
    s_img.SetSpacing(spacing)
    s_msk.SetSpacing(spacing)
    a = rj.extract_radiomic_features(s_img, s_msk, n_bins=16)
    b = rj.extract_radiomic_features(img, msk, spacing, n_bins=16)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and not k.startswith("diagnosis_"):
            assert a[k] == pytest.approx(b[k], nan_ok=True), k


def test_bad_inputs(cube):
    img, msk = cube
    with pytest.raises(TypeError):
        rj.extract_radiomic_features("a.nii.gz", msk, [1, 1, 1])
    with pytest.raises(ValueError):
        rj.extract_radiomic_features(img[None], msk[None], [1, 1, 1])
    with pytest.raises(ValueError):
        rj.extract_radiomic_features(img, msk, [1, 1, 1], slices_2d=[(3, 4)])
