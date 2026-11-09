# radiomicsjl

Python interface to [Radiomics.jl](https://github.com/pzaffino/Radiomics.jl): fast, IBSI-compliant radiomic feature extraction, on CPU or NVIDIA GPU.

```bash
pip install radiomicsjl
```

No Julia installation is needed. The first call downloads Julia and Radiomics.jl into the
current Python environment (a few minutes, once); after that, start-up takes a few seconds.

## Usage

```python
import radiomicsjl as rj

# From files (NIfTI, NRRD, DICOM file or folder): spacing is read from the header
feats = rj.extract_radiomic_features("ct.nii.gz", "mask.nii.gz", n_bins=32)

# From numpy arrays: spacing is required, one value per array axis
feats = rj.extract_radiomic_features(image, mask, spacing=[0.98, 0.98, 3.0], bin_width=25)

print(feats["firstorder_mean"], feats["glcm_Contrast"])
```

The arguments are the same as `Radiomics.extract_radiomic_features` in Julia:

| argument | default | |
|---|---|---|
| `features` | all | e.g. `["first_order", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape3d"]` (`"shape2d"` for 2D images) |
| `labels` | `1` | an int, or a list of ints → `{label: features}` |
| `n_bins` / `bin_width` | 32 bins | discretisation |
| `weighting_norm` | `None` | e.g. `"euclidean"` |
| `keep_largest_only` | `True` | keep only the largest connected component |
| `slices_2d` | `None` | list of `(axis, index)`, **0-based** as in numpy: `(2, 10)` is `image[:, :, 10]` → `{(axis, index): features}` |
| `features_std` | `False` | per-direction standard deviation for GLCM/GLRLM |
| `get_raw_matrices` | `False` | also return the texture matrices (numpy arrays) |
| `use_gpu` | `False` | CUDA acceleration (see below) |
| `suv_ibsi_normalization` | `False` | SUVbw normalisation for PET DICOM |
| `verbose` | `False` | |

### SimpleITK, numpy and spacing

The easiest way with SimpleITK is to pass the images themselves: array and spacing are taken
from them, in (x, y, z) order like the file readers.

```python
import SimpleITK as sitk
img, msk = sitk.ReadImage("ct.nii.gz"), sitk.ReadImage("mask.nii.gz")
feats = rj.extract_radiomic_features(img, msk, n_bins=32)
```

With numpy arrays, `spacing` must follow the array axes. Beware that
`sitk.GetArrayFromImage` returns (z, y, x) while `GetSpacing()` is (x, y, z):
passing them together gives wrong shape and texture features. Pass the image instead,
or reverse the spacing (`GetSpacing()[::-1]`). Arrays from nibabel are already (x, y, z),
in the same order as `header.get_zooms()`.

## GPU

```bash
python -m radiomicsjl install-gpu   # adds CUDA.jl to the Julia environment (once)
```

then `use_gpu=True`. Requires an NVIDIA GPU with a recent driver.

## Other commands

```bash
python -m radiomicsjl setup     # install Julia + Radiomics.jl now instead of at first use
python -m radiomicsjl version
```

Julia uses all CPU cores by default; set `PYTHON_JULIACALL_THREADS=4` (before importing) to limit it.
The Julia environment lives inside the Python environment, so each virtualenv/conda env has its own.

## License

MIT
