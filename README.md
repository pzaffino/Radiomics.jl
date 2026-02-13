# **Welcome to Radiomics.jl**
<a href="https://github.com/pzaffino/radiomics.jl">
  <img src="https://raw.githubusercontent.com/pzaffino/Radiomics.jl/refs/heads/main/Logo%20Radiomicsjl.png" alt="Logo Radiomicsjl" width="500"/>
</a>


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

The signature of the main function is:
```julia
    extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                              features=Symbol[],
                              labels=nothing,
                              n_bins=nothing,
                              bin_width=nothing,
                              weighting_norm=nothing,
                              force_2d::Bool=false,
                              force_2d_dimension::Int=1,
                              keep_largest_only::Bool=true,
                              sample_rate=0.03,
                              verbose::Bool=false)
        
    # Parameters:
    - `img_input`: The input image (Array).
    - `mask_input`: The mask defining the region of interest (Array).
    - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
    - `features`: Array of symbols specifying which features to compute. 
                 Options: :first_order, :glcm, :shape2d, :shape3d, :glszm, :ngtdm, :glrlm, :gldm.
    - `labels`: Single label (Int), multiple labels (Vector{Int}), or nothing for default (label 1).
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `weighting_norm`: Performs weight-normalized radiomic feature extraction on the input image and mask (optional).
                        Options: "infinity", "euclidean", "manhattan", and "no_weighting".
    - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
    - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
    - `keep_largest_only`: If true, keeps only the largest connected component for 3D shape features (default: true).
    - `sample_rate`: The sample rate for feature extraction (optional).
    - `verbose`: If true, prints progress messages.
        
    # Returns:
    - Single label or nothing: Dict{String,Any} with feature names as keys
    - Multiple labels: Dict{Int,Dict{String,Any}} where outer keys are label values
```

### Specific cases

To compute only a subset of features, specify the desired ones using the features flags:
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features = [:glcm, :gldm])
```

To extract features from a specific label (default is 1), execute:

```julia
radiomic_features = Radiomics.extract_radiomic_features(ct, mask, spacing, labels=4)
```

To extract features just from a list of labels, execute:

```julia
radiomic_features = Radiomics.extract_radiomic_features(ct, mask, spacing, labels=[3, 5, 11])
```
Please note that, in this case, the function returns a dictionary with integer keys (e.g., 3, 5, and 11).

To compute more features with a specific bin_width (by default it is 25.0):
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm, :glszm], bin_width=35.0);
```

To compute with a specific number of bins:
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm, :glszm], n_bins=16);
```

To compute features from the entire mask regardless of fragmentation, set keep_largest_only to false; set it to true to isolate and analyze only the largest connected component
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false);
```

To compute with a specific weighting_norm 
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false, weighting_norm="euclidean");
```

## Multi threading
Radiomics.jl can be run in multi-threading mode (highly recommended to speed up the computation).

To do this, you can define and set the following environment variable: JULIA_NUM_THREADS=auto

On a Linux system, the .bashrc file can be edited by adding the line:

```bash
export JULIA_NUM_THREADS=auto
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

To update the Radiomics.jl package, execute:

```python
from juliacall import Main as jl
jl.seval('import Pkg; Pkg.update("Radiomics")')
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

### Specific cases

If you want to extract only one subset of features, you can use:

```python
radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing, features="first_order")
```

To extract two or more subsets (e.g. first order and glcm), you can run:

```python
radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing, features=["first_order", "glcm"])
```

To extract features from a specific label (default is 1), execute:

```python
radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing, labels=4)
```

To extract features just from a list of labels, execute:

```python
radiomic_features = jl.Radiomics.extract_radiomic_features(ct, mask, spacing, labels=[3, 5, 11])
```
Please note that, in this case, the function returns a dictionary with integer keys (e.g., 3, 5, and 11).

# Generate C shared library (and use it in Python and C++)

It is also possible to generate a C shared library (.dll, .so, or .dylib) and call it from C/C++ code or any language that provides a C shared library interface.

To generate the library, navigate to the Radiomics.jl source folder and run the following in Julia (this will take a few minutes):

```julia
using PackageCompiler

create_library(".", "radiomicsjl_build";
               lib_name="libradiomicsjl",
               force=true,
               incremental=true,
               filter_stdlibs=true)
```

The folder radiomicsjl_build will contain the shared libraries.

## Call the C shared library from Python

For example, to extract features in Python using the shared library (in a Linux environment), run:

```python
import ctypes
import os
import json
import numpy as np
import SimpleITK as sitk

lib_path = os.path.abspath("SHARED_LIB_PATH/radiomicsjl_build/lib/libradiomicsjl.so")
lib = ctypes.CDLL(lib_path)
lib.init_julia(0, None)

lib.c_extract_radiomic_features.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # img_ptr   (Float32)
    ctypes.c_int64,                  # size_x    (Int64)
    ctypes.c_int64,                  # size_y    (Int64)
    ctypes.c_int64,                  # size_z    (Int64)
    ctypes.POINTER(ctypes.c_float),  # mask_ptr  (Float32)
    ctypes.c_double,                 # spacing_x (Float64)
    ctypes.c_double,                 # spacing_y (Float64)
    ctypes.c_double,                 # spacing_z (Float64)
    ctypes.c_int64                   # n_bins    (Int64)
]

lib.c_extract_radiomic_features.restype = ctypes.c_char_p

ct_sitk = sitk.ReadImage('DATA_PATH/CT.nrrd')
mask_sitk = sitk.ReadImage('DATA_PATH/left_parotid.nrrd')

ct = np.asfortranarray(sitk.GetArrayFromImage(ct_sitk), dtype=np.float32)
mask = np.asfortranarray(sitk.GetArrayFromImage(mask_sitk), dtype=np.float32)

ptr_ct = ct.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
ptr_mask = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

sx, sy, sz = ct.shape
spacing = ct_sitk.GetSpacing()
bin_width = 25

raw_json = lib.c_extract_radiomic_features(
        ptr_ct,
        sx, sy, sz,
        ptr_mask,
        float(spacing[0]), float(spacing[1]), float(spacing[2]),
        int(bin_width))

readiomic_features = json.loads(raw_json.decode('utf-8'))
```

## Call the C shared library from C++

Here is a C++ code snippet to extract radiomic features using the shared library on Linux:

`CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.10)
project(Radiomicsjl)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(Radiomicsjl main.cpp)

# Link ITK and system library 'dl' to load .so file
target_link_libraries(Radiomicsjl 
    ${ITK_LIBRARIES}
    dl
)
```

`main.cpp`

```cpp
#include "itkImage.h"
#include "itkImageFileReader.h"
#include <iostream>
#include <vector>
#include <dlfcn.h> // to load .so file in Linux/macOS

// Define function signature of the function in the .so file
typedef const char* (*ExtractFeaturesFunc)(
    float*,                      // img_ptr (c_float)
    int64_t, int64_t, int64_t,   // size_x, size_y, size_z (c_int64)
    float*,                      // mask_ptr (c_float)
    double, double, double,      // spacing_x, spacing_y, spacing_z (c_double)
    int64_t                      // n_bins (c_int64)
);


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " anatomical_image binary_mask  bin_width" << std::endl;
        return 1;
    }

    // Get parameters from command line
    std::string imagePath = argv[1];
    std::string maskPath = argv[2];
    double binWidth = std::stod(argv[3]);

    // Read images 
    using PixelType = float;
    using ImageType = itk::Image<PixelType, 3>;
    using ReaderType = itk::ImageFileReader<ImageType>;

    auto readerImg = ReaderType::New();
    readerImg->SetFileName(imagePath);
    
    auto readerMask = ReaderType::New();
    readerMask->SetFileName(maskPath);

    try {
        readerImg->Update();
        readerMask->Update();
    } catch (itk::ExceptionObject &ex) {
        std::cerr << "ITK error: " << ex.GetDescription() << std::endl;
        return 1;
    }

    ImageType::Pointer img = readerImg->GetOutput();
    ImageType::Pointer mask = readerMask->GetOutput();

    // Prepare arguments for the shared library call
    auto size = img->GetLargestPossibleRegion().GetSize();
    int64_t nx = size[0];
    int64_t ny = size[1];
    int64_t nz = size[2];

    auto spacingITK = img->GetSpacing();
    double sx = spacingITK[0];
    double sy = spacingITK[1];
    double sz = spacingITK[2];

    float* imgPtr = img->GetBufferPointer();
    float* maskPtr = mask->GetBufferPointer();

    int64_t nBins = static_cast<int64_t>(binWidth);

    // Load and initialize shared library
    void* handle = dlopen("SHARED_LIB_PATH/radiomicsjl_build/lib/libradiomicsjl.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error during loading the library: " << dlerror() << std::endl;
        return 1;
    }

    typedef void (*InitJuliaFunc)(int, char**);

    auto init_julia = (InitJuliaFunc) dlsym(handle, "init_julia");
    if (init_julia) {
        init_julia(0, NULL);
    }

    auto extract_features = (ExtractFeaturesFunc) dlsym(handle, "c_extract_radiomic_features");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Failed to find function symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    // Run feature extraction
    std::cout << "Running feature extraction with bin_width = " << binWidth << "..." << std::endl;
    const char* json_result = extract_features(
        imgPtr,
        nx, ny, nz,
        maskPtr,
        sx, sy, sz,
        nBins); 

    if (json_result) {
        std::cout << "Features (JSON):\n" << json_result << std::endl;
    } else {
        std::cerr << "The function returned a null pointer!" << std::endl;
    }

    dlclose(handle);
    return 0;
}
```

## **Website with complete documentation**
For complete documentation, visit the [official website](https://www.radiomicsjl.org).

