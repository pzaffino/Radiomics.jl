# **Welcome to Radiomics.jl**
<a href="https://github.com/pzaffino/radiomics.jl">
  <img src="https://raw.githubusercontent.com/pzaffino/Radiomics.jl/refs/heads/main/Logo%20Radiomicsjl.png" alt="Logo Radiomicsjl" width="500"/>
</a>


**Radiomics.jl** is an open-source Julia library for extracting radiomics features from medical images.

:warning: **THE LIBRARY IS CURRENTLY IN AN EARLY STAGE AND UNDER HEAVY DEVELOPMENT** :warning:

:email: If you're interested in contributing to the project, please contact us via email at "p DOT zaffino AT unicz DOT it" :email:


## **Implemented features**

Right now, the implemented features are:
- first order features
- 2D shape features
- 3D shape features
- GLSZM features
- NGTDM features
- GLRLM features

Additional features (e.g. texture) have not been implemented yet, but we are on the right path!

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

radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose = true)
```
