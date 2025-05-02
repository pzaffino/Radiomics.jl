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

Additional features (e.g. shape and texture) have not been implemented yet, but we are on the right path!

## **Example**

```julia
using NIfTI
using Radiomics

ct = niread("sample_data/CTChest.nii.gz");
mask = niread("sample_data/Lungs.nii.gz");

radiomic_features = Radiomics.extract_radiomic_features(ct, mask; binarize_mask = true, verbose = false);
```
