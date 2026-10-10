using DICOM
using NRRD
using NIfTI

using LinearAlgebra: Diagonal

"""
    MedImage{T,N}
    
    A medical image with the given type T and number of dimensions N.
    
    # Parameters:
    - `T`: Type of the image data.
    - `N`: Number of dimensions of the image data.
    
    # Fields:
    - `data`: The image data.
    - `spacing`: The spacing of the voxels.
    - `origin`: The origin of the image.
    - `direction`: The direction of the image.
"""
struct MedImage{T,N}
    data::Array{T,N}
    spacing::Vector{Float64}
    origin::Vector{Float64}
    direction::Matrix{Float64}
end

"""
    read_image(path)
    
    Reads an image from the given path.
    
    # Parameters:
    - `path`: Path to the image.
    
    # Returns:
    - `MedImage`: The image.
"""
function read_image(path::AbstractString)
    p = lowercase(path)
    if endswith(p, ".nii") || endswith(p, ".nii.gz")
        return _read_nifti(path)
    end
    error("Not supported format: $path")
end

"""
    _resolve_inputs(img, mask, spacing)
    
    Resolves the inputs to the given function.
    
    # Parameters:
    - `img`: The input image.
    - `mask`: The input mask.
    - `spacing`: The spacing of the voxels.
    
    # Returns:
    - `Tuple{Array{T,N},Array{T,N},Vector{Float64}}`: The resolved inputs.
"""
function _resolve_inputs(img, mask, spacing)
    img_is_path  = img  isa AbstractString
    mask_is_path = mask isa AbstractString

    img_is_path == mask_is_path ||
        error("Image and mask must be both paths or both arrays")

    if img_is_path
        i = read_image(img)
        m = read_image(mask)
        all(isapprox.(i.spacing, m.spacing; atol=1e-3)) ||
            error("Different Spacing: image $(i.spacing), mask $(m.spacing)")
        return i.data, m.data, isnothing(spacing) ? i.spacing : spacing
    end

    isnothing(spacing) && error("With arrays, spacing is required")
    return img, mask, spacing
end

"""
    _read_nifti(path)
    
    Reads a NIfTI image from the given path.
    
    # Parameters:
    - `path`: Path to the NIfTI image.
    
    # Returns:
    - `MedImage`: The NIfTI image.
"""
function _read_nifti(path)
    vol = NIfTI.niread(path)
    data = Array(vol.raw)
    N = ndims(data)
    sp = collect(Float64.(NIfTI.voxel_size(vol.header)))[1:N]
    return MedImage(data, sp, zeros(N), Matrix(Diagonal(ones(N))))
end