"""radiomicsjl — Python interface to Radiomics.jl.

    >>> import radiomicsjl
    >>> feats = radiomicsjl.extract_radiomic_features("ct.nii.gz", "mask.nii.gz", n_bins=32)

Julia and Radiomics.jl are installed automatically the first time a function is called.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Iterable, Sequence, Union

import numpy as np

from ._julia import get_bridge, install_gpu

try:
    __version__ = _pkg_version("radiomicsjl")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = ["extract_radiomic_features", "install_gpu", "julia_version", "__version__"]

PathLike = Union[str, os.PathLike]
ArrayOrPath = Union[np.ndarray, PathLike, Any]  # Any: SimpleITK.Image


def _kind(x) -> str:
    if isinstance(x, (str, os.PathLike)):
        return "path"
    if _is_sitk(x):
        return "SimpleITK image"
    return "array"


def _is_sitk(x) -> bool:
    return type(x).__module__.startswith("SimpleITK") and hasattr(x, "GetSpacing")


def _from_sitk(img):
    """SimpleITK image -> (array indexed (x, y, z), spacing (x, y, z)), same order as the file readers."""
    import SimpleITK as sitk

    arr = sitk.GetArrayViewFromImage(img).transpose()  # (z, y, x) -> (x, y, z), no copy
    return arr, tuple(float(v) for v in img.GetSpacing())


def _image_input(x: ArrayOrPath, name: str):
    if isinstance(x, (str, os.PathLike)):
        return os.fspath(x)
    a = np.asarray(x)
    if a.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D or 3D array, got {a.ndim}D")
    if a.dtype == object or not (np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_):
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    # native byte order and a single memory block (C or Fortran order): safe and fast to read from Julia
    a = a.astype(a.dtype.newbyteorder("="), copy=False)
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        a = np.ascontiguousarray(a)
    return a


def _features_input(features: Union[None, str, Iterable[str]]) -> str:
    if features is None:
        return ""
    if isinstance(features, str):
        return "" if features.lower() == "all" else features.lower()
    return ",".join(str(f).lower().lstrip(":") for f in features)


def extract_radiomic_features(
    image: ArrayOrPath,
    mask: ArrayOrPath,
    spacing: Union[None, Sequence[float]] = None,
    *,
    features: Union[None, str, Iterable[str]] = None,
    labels: Union[None, int, Sequence[int]] = None,
    n_bins: Union[None, int] = None,
    bin_width: Union[None, float] = None,
    weighting_norm: Union[None, str] = None,
    keep_largest_only: bool = True,
    get_raw_matrices: bool = False,
    features_std: bool = False,
    slices_2d: Union[None, Sequence[Sequence[int]]] = None,
    use_gpu: bool = False,
    suv_ibsi_normalization: bool = False,
    verbose: bool = False,
) -> dict[Any, Any]:
    """Extract radiomic features with Radiomics.jl.

    Same arguments and defaults as ``Radiomics.extract_radiomic_features`` in Julia.

    Parameters
    ----------
    image, mask
        Both file paths (NIfTI, NRRD, DICOM file/folder), both ``SimpleITK.Image``,
        or both 2D/3D numpy arrays.
        Arrays are used with numpy indexing: ``image[i, j, k]`` is the voxel at Julia
        index ``[i+1, j+1, k+1]``.
        SimpleITK images are used in (x, y, z) order with their own spacing, exactly
        like the file readers, so axis 2 is the slice axis.
    spacing
        Voxel spacing, one value per array axis, in the same order as the array axes.
        Required for arrays; read from the header/image if omitted for files and
        SimpleITK images. Do not pass ``GetSpacing()`` together with an array from
        ``sitk.GetArrayFromImage`` (that array is (z, y, x)): pass the image itself.
    features
        ``None``/"all", or a list such as ``["glcm", "first_order", "shape3d"]``.
    labels
        Label value or list of label values (default: 1).
    slices_2d
        List of ``(axis, index)`` pairs, 0-based as in numpy: ``(2, 10)`` is
        ``image[:, :, 10]``. Several pairs return ``{(axis, index): {...}}``.

    Returns
    -------
    dict
        ``{feature_name: value}`` for one label and one slice; ``{label: {...}}`` for
        several labels; ``{(axis, index): {...}}`` for several slices.
    """
    if _kind(image) != _kind(mask):
        raise TypeError(f"image and mask must be of the same kind, got {_kind(image)} and {_kind(mask)}")

    if _is_sitk(image):
        image, img_spacing = _from_sitk(image)
        mask, msk_spacing = _from_sitk(mask)
        if not np.allclose(img_spacing, msk_spacing, atol=1e-3):
            raise ValueError(f"Different spacing: image {img_spacing}, mask {msk_spacing}")
        if spacing is None:
            spacing = img_spacing

    spacing_in = None if spacing is None else np.asarray(spacing, dtype=np.float64).ravel()

    if labels is None or isinstance(labels, (int, np.integer)):
        labels_in = None if labels is None else int(labels)
    else:
        labels_in = np.asarray(labels, dtype=np.int64).ravel()

    slices_in = None
    if slices_2d is not None:
        s = np.asarray(slices_2d, dtype=np.int64).reshape(-1, 2)
        if (s < 0).any() or (s[:, 0] > 2).any():
            raise ValueError("slices_2d must contain 0-based (axis, index) pairs with axis in 0..2")
        slices_in = s + 1  # Julia is 1-based

    image_in = _image_input(image, "image")
    mask_in = _image_input(mask, "mask")
    if not isinstance(mask_in, str) and np.issubdtype(mask_in.dtype, np.floating):
        mask_in = np.rint(mask_in).astype(np.int64)
    features_in = _features_input(features)

    result = get_bridge().extract(
        image_in,
        mask_in,
        spacing_in,
        features_in,
        labels_in,
        slices_in,
        None if n_bins is None else int(n_bins),
        None if bin_width is None else float(bin_width),
        None if weighting_norm is None else str(weighting_norm),
        bool(keep_largest_only),
        bool(get_raw_matrices),
        bool(features_std),
        bool(use_gpu),
        bool(suv_ibsi_normalization),
        bool(verbose),
    )

    # Several slices: keys (plane, slice) come back 1-based from Julia -> 0-based
    if slices_in is not None and len(slices_in) > 1:
        result = {(p - 1, i - 1): v for (p, i), v in result.items()}
    return result


def julia_version() -> str:
    """Version of the Radiomics.jl package in use."""
    return str(get_bridge().version())