"""radiomicsjl — Python interface to Radiomics.jl.

    >>> import radiomicsjl
    >>> feats = radiomicsjl.extract_radiomic_features("ct.nii.gz", "mask.nii.gz", n_bins=32)

Julia and Radiomics.jl are installed automatically the first time a function is called.
"""

from __future__ import annotations

import json
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


def _julia_literal(v, name: str) -> str:
    """Python value -> Julia literal (numbers, booleans, strings, lists/tuples of them)."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f:
            return "NaN"
        if f in (float("inf"), float("-inf")):
            return "Inf" if f > 0 else "-Inf"
        return repr(f)
    if isinstance(v, str):
        return json.dumps(v).replace("$", "\\$")
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_julia_literal(x, name) for x in v) + "]"
    raise TypeError(f"filter_params[{name!r}]: unsupported value {v!r} ({type(v).__name__}); "
                    "use numbers, booleans, strings or lists of them")


def _filter_params_input(params) -> str:
    if params is None:
        return ""
    if not isinstance(params, dict):
        raise TypeError(f"filter_params must be a dict, e.g. {{'level': 2}}, got {type(params).__name__}")
    items = []
    for k, v in params.items():
        if not (isinstance(k, str) and k.isidentifier()):
            raise ValueError(f"filter_params: invalid parameter name {k!r}")
        items.append(f"{k}={_julia_literal(v, k)}")
    return "(" + ", ".join(items) + ",)" if items else ""


def _filter_type_input(filter_type) -> str:
    if filter_type is None:
        return ""
    names = [filter_type] if isinstance(filter_type, str) else list(filter_type)
    if not names or not all(isinstance(n, str) and n.strip() and "," not in n for n in names):
        raise ValueError(f"filter_type must be a filter name or a list of names, got {filter_type!r}")
    return ",".join(n.strip() for n in names)


def _slices_to_0_based(obj):
    """Recursively turn the 1-based (plane, slice) keys returned by Julia into 0-based ones."""
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
            k = (k[0] - 1, k[1] - 1)
        out[k] = _slices_to_0_based(v)
    return out


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
    filter_type: Union[None, str, Sequence[str]] = None,
    filter_params: Union[None, dict] = None,
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
    filter_type
        Image filter applied before extraction: ``"haar"``, ``"log"``, ``"gabor"``, or a list.
        Features are computed on every response map; with more than one map the result is
        ``{"<filter>_<map>": {...}}`` (e.g. ``"haar_wavelet-LLH"``). Shape features are not
        computed on filtered images.
    filter_params
        Filter parameters as a dict, e.g. ``{"level": 2, "subbands": ["LLL", "HHH"]}`` (Haar),
        ``{"sigma": [1.0, 3.0]}`` (LoG, mm), ``{"sigma": 5.0, "lambda": 2.0}`` (Gabor).
        The same parameters are passed to every filter in ``filter_type``.

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
        # e.g. masks stored as UInt8 0/255 with scl_slope 1/255: same rounding as Radiomics.jl's readers
        mask_in = np.rint(mask_in).astype(np.int64)
    features_in = _features_input(features)
    filter_type_in = _filter_type_input(filter_type)
    filter_params_in = _filter_params_input(filter_params)

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
        filter_type_in,
        filter_params_in,
        bool(verbose),
    )

    # Several slices: keys (plane, slice) come back 1-based from Julia -> 0-based,
    # also when nested under filter response maps
    if slices_in is not None and len(slices_in) > 1:
        result = _slices_to_0_based(result)
    return result


def julia_version() -> str:
    """Version of the Radiomics.jl package in use."""
    return str(get_bridge().version())
