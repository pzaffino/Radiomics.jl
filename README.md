# **Welcome to Radiomics.jl**
<div align=center>
  <a href="https://github.com/pzaffino/radiomics.jl">
  <img src="https://raw.githubusercontent.com/pzaffino/Radiomics.jl/refs/heads/main/Logo%20Radiomicsjl.png" alt="Logo Radiomicsjl" width="500"/>
</a>
</div>
</br>

**Radiomics.jl** is an open-source, [IBSI](https://theibsi.github.io/) compliant Julia library for extracting radiomics features from medical images. 

You can integrate the library with your preferred programming language: Radiomics.jl is also  **compatible with Python, C++, R, and many others**!

For complete documentation, visit the [official website](https://www.radiomicsjl.org).

📧 If you're interested in contributing to the project, please contact us via email at <a href="mailto:p.zaffino&#64;unicz.it">p.zaffino@unicz.it</a>.




- [Welcome to Radiomics.jl](#welcome-to-radiomicsjl)
  - [Implemented features](#implemented-features)
  - [Why choose Radiomics.jl for your project](#why-choose-radiomicsjl-for-your-project)
- [Getting started](#getting-started)
  - [Example](#example)
- [Multi threading](#multi-threading)
- [Using Radiomics.jl from Python](#using-radiomicsjl-from-python)
  - [Setup](#setup)
  - [Feature extraction](#feature-extraction)
- [Discalaimer](#discalaimer)

## **Implemented features**
Right now, the implemented features are:
- first-order features
- 2D shape features
- 3D shape features
- GLCM features
- GLDM features
- GLRLM features
- GLSZM features
- NGTDM features

## **Why choose Radiomics.jl for your project**
- **Full IBSI Compliance**: Every feature is calculated according to IBSI 1 - Phase 1 standards—our unit tests are strictly validated against IBSI sample images (go ahead, take a look!).
- **GPU acceleration**: Harness native NVIDIA GPU acceleration powered by CUDA.jl for feature extraction, with seamless, automatic CPU fallback if no GPU is available.
- **User-Friendly API:** Simplify your workflow by extracting features from multiple labels with a single call. Handling 2D slices or multiple planes/slices "in one shot" has never been easier.
- **First-class 2D support:** 2D analysis is treated with the same priority as 3D; computing features for a 2D matrix is just as seamless as the 3D case.
- **Deep Insights:** While other radiomics libraries focus only on final scalar values, Radiomics.jl also lets you easily get the raw matrices used under the hood for texture feature calculations.
- **No silent failures:** If something looks suspicious or ambiguous—like disconnected segmentation islands or oversized binning—the library raises clear warnings.
- **Native Multi-threading:** The engine is engineered from the ground up to take full advantage of multi-threaded performance.
- **Cross-language ready:** Radiomics.jl delivers a frictionless experience across languages, allowing Python, C++, R, and many other language developers to call Julia’s power with ease.

# **Getting started**
To install Radiomics.jl, simply run:

```julia
import Pkg
Pkg.add("Radiomics")
```
You can install Radiomics.jl on either your local machine or Google Colab.

## **Example**
Once the library is installed, radiomics features can be extracted as reported in the following example (sample data shipped with the library are used):

```julia
using NIfTI
using Radiomics

ct = niread("sample_data/CTChest.nii.gz")
mask = niread("sample_data/Lungs.nii.gz")
spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]

radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing)
```
# Using Radiomics.jl with CUDA
To use Radiomics.jl with GPU acceleration, you need to install CUDA.
```julia
import Pkg
Pkg.add("CUDA")
```
Once the library is installed, radiomics features can be extracted as reported in the following example:
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing, use_gpu = true)
```
To enable CUDA streams for the concurrent extraction of feature families, you need to set the flag as shown in the following example:
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing, use_gpu = true)
```
# Multi threading
Radiomics.jl can be run in multi-threading mode (highly recommended to speed up the computation).

To do this, you can define and set the following environment variable: JULIA_NUM_THREADS=auto

On a Linux system, the .bashrc file can be edited by adding the line:

```bash
export JULIA_NUM_THREADS=auto
```

If you want to use Radiomics.jl from Python by using Juliacall (see below), add also this line:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
```

# Using Radiomics.jl from Python
Radiomics.jl can be used directly from Python through the `radiomicsjl` package.

## Setup

```bash
pip install radiomicsjl
```

Julia and Radiomics.jl are installed automatically the first time a feature extraction is run (a few minutes, only once). If a compatible Julia is already installed on your system, it is used; otherwise an embedded version is downloaded.

For GPU acceleration (NVIDIA GPU required), install CUDA.jl once:

```bash
python -m radiomicsjl install-gpu
```

## Feature extraction

From image files (NIfTI, NRRD, DICOM file or folder), with the voxel spacing read from the header:

```python
import radiomicsjl as rjl

radiomic_features = rjl.extract_radiomic_features('DATA_PATH/ct.nii.gz', 'DATA_PATH/mask.nii.gz')
```

From SimpleITK images:

```python
import SimpleITK as sitk
import radiomicsjl as rjl

ct_sitk = sitk.ReadImage('DATA_PATH/ct.nii.gz')
mask_sitk = sitk.ReadImage('DATA_PATH/mask.nii.gz')

radiomic_features = rjl.extract_radiomic_features(ct_sitk, mask_sitk)
```

From numpy arrays, with the spacing in the same order as the array axes:

```python
ct = sitk.GetArrayFromImage(ct_sitk)        # axes (z, y, x)
mask = sitk.GetArrayFromImage(mask_sitk)
spacing = ct_sitk.GetSpacing()[::-1]        # GetSpacing() is (x, y, z): reverse it

radiomic_features = rjl.extract_radiomic_features(ct, mask, spacing)
```

The arguments are the same as in Julia (`features`, `labels`, `n_bins`, `bin_width`, `use_gpu`, ...). `slices_2d` is 0-based in Python: `(2, 10)` is the slice `[:, :, 10]`.

## Using juliacall directly

Radiomics.jl can also be called through [juliacall](https://github.com/JuliaPy/PythonCall.jl) without the `radiomicsjl` package:

```bash
pip install juliacall
```

```python
from juliacall import Main as jl
jl.seval('import Pkg; Pkg.add("Radiomics")')
jl.seval("using Radiomics")

ct = sitk.GetArrayFromImage(ct_sitk)
mask = sitk.GetArrayFromImage(mask_sitk)
spacing = ct_sitk.GetSpacing()[::-1]        # match the (z, y, x) array axes

radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing)
```
## Using Radiomics.jl from Docker

### Install

Image with CUDA (NVIDIA GPU) acceleration:

    docker pull ghcr.io/pzaffino/radiomics.jl:latest-gpu

Image without GPU acceleration:

    docker pull ghcr.io/pzaffino/radiomics.jl:latest

### Extract features

Mount the folder with your data as `/data` and pass the image and the mask (NIfTI, NRRD or DICOM folder):

    docker run --rm -v $(pwd):/data ghcr.io/pzaffino/radiomics.jl:latest /data/ct.nii.gz /data/mask.nii.gz -o /data/features.csv

On GPU (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

    docker run --rm --gpus all -v $(pwd):/data ghcr.io/pzaffino/radiomics.jl:latest-gpu /data/ct.nii.gz /data/mask.nii.gz -o /data/features.csv

Use `--gpus '"device=0"'` to select a specific GPU.

### Options

| option | example | description |
|---|---|---|
| `-o`, `--output` | `/data/features.csv` | output CSV file |
| `--features` | `first_order,glcm,shape3d` | feature families (default: all) |
| `--labels` | `1` or `1,2,3` | label values (default: 1) |
| `--n_bins` / `--bin_width` | `32` / `25` | discretisation |
| `--weighting_norm` | `euclidean` | weighting normalisation |
| `--keep_largest_only` | `false` | keep only the largest connected component (default: true) |
| `--slices_2d` | `3:10,3:11` | plane:slice pairs, 1-based |
| `--filter_type` | `haar`, `log`, `gabor` | image filter applied before extraction |
| `--filter_params` | `'(level=2,)'`, `'(sigma=[1.0, 3.0],)'` | filter parameters (Julia NamedTuple) |
| `--features_std` | `true` | std, min and max over directions for GLCM and GLRLM |
| `--suv_ibsi_normalization` | `true` | SUV normalisation of a PET DICOM series |
| `--use_gpu` | `false` | GPU on/off (default: on in the `-gpu` image) |
| `--verbose` | `false` | progress messages (default: true) |

With several labels, slices or filter response maps, the CSV has one row each, identified by the `label`, `slice` and `filter_map` columns.

Example with the Haar wavelet:

    docker run --rm -v $(pwd):/data ghcr.io/pzaffino/radiomics.jl:latest /data/ct.nii.gz /data/mask.nii.gz --filter_type haar --features first_order,glcm -o /data/features_haar.csv

Multi-threading:

    docker run --rm -e JULIA_NUM_THREADS=auto -v $(pwd):/data ghcr.io/pzaffino/radiomics.jl:latest /data/ct.nii.gz /data/mask.nii.gz

# Discalaimer
This software is for research purposes only.
It is NOT intended for clinical use, diagnosis, or treatment.
It has NOT been cleared or approved by the FDA (USA), CE (EU), or any other regulatory body.
Use at your own risk.

# How cite Radiomics.jl
If you use Radiomics.jl in your research, please cite it as:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23160114.svg)](https://doi.org/10.5281/zenodo.23160114)
For detailed citation formats and BibTeX entries, see our [Citation Guide](https://pzaffino.github.io/Radiomics.jl/cite.html).
