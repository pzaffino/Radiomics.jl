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
To compute only a subset of features, specify the desired ones using the features flag.
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features = [:glcm, :gldm])
```
To compute more features with a specific bin_width
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm, :glszm], bin_width=25.0f0);
```
To compute with a specific number of bins
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm, :glszm], n_bins=16);
```
To compute features from the entire mask regardless of fragmentation, set keep_largest_only to false; set it to true to isolate and analyze only the largest connected component
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false);
```
To compute with weighting_norm personalzed 
```julia
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false, weighting_norm="euclidean");
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

spacing = list(ct_sitk.GetSpacing())

radiomic_features = dict(jl.Radiomics.extract_radiomic_features(ct, mask, spacing))
```
# Generate C shared library

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
    void* handle = dlopen("/home/eica2026/Radiomics.jl/radiomicsjl_build/lib/libradiomicsjl.so", RTLD_LAZY);
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

