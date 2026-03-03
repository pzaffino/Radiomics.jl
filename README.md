# **Welcome to Radiomics.jl**
<a href="https://github.com/pzaffino/radiomics.jl">
  <img src="https://raw.githubusercontent.com/pzaffino/Radiomics.jl/refs/heads/main/Logo%20Radiomicsjl.png" alt="Logo Radiomicsjl" width="500"/>
</a>

## **Website with complete documentation**
For complete documentation, visit the [official website](https://www.radiomicsjl.org).

**Radiomics.jl** is an open-source Julia library for extracting radiomics features from medical images.

:email: If you're interested in contributing to the project, please contact us via email at "p DOT zaffino AT unicz DOT it" :email:

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

## **Getting started**

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

## Using Radiomics.jl from Python
Radiomics.jl can be used directly from Python in a simple and convenient way.

### Setup

Juliacall acts as a bridge between Python and Julia.

Install it via:

If Julia is not already installed on your system, juliacall will download and use its own embedded version.

```bash
pip install juliacall
```

To install the Radiomics.jl package, run the following in Python:

```python
from juliacall import Main as jl
jl.seval('import Pkg; Pkg.add("Radiomics")')
```

### Feature extraction

Once the environment is set up, you can extract radiomic features as shown below:

```python
import SimpleITK as sitk
import numpy as np

from juliacall import Main as jl
jl.seval("using Radiomics")

ct_sitk = sitk.ReadImage('DATA_PATH/ct.nii.gz')
mask_sitk = sitk.ReadImage('DATA_PATH/mask.nii.gz')

ct = sitk.GetArrayFromImage(ct_sitk)
mask = sitk.GetArrayFromImage(mask_sitk)

spacing = ct_sitk.GetSpacing()

radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing)
```
