# **Welcome to Radiomics.jl**


**Radiomics.jl** is an open-source Julia library for extracting radiomics features from medical images.

:warning: :warning: :warning: **THE LIBRARY IS CURRENTLY IN AN EARLY STAGE AND UNDER HEAVY DEVELOPMENT** :warning: :warning: :warning:

Several features have not been implemented yet, but we are on the right path!

If you want to contribute to the project, feel absolutely free to get in touch! :email:


## **Example**

```
using NIfTI
using Radiomics

ct = niread("sample_data/CTChest.nii.gz");
mask = niread("sample_data/Lungs.nii.gz");

radiomic_features = Radiomics.extract_radiomic_features(ct, mask; binarize_mask = true, verbose = false);
```
