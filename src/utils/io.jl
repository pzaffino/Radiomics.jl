using DICOM
using NRRD
using NIfTI

using LinearAlgebra: Diagonal, cross, dot

include("utils_dicom.jl")

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
    
    Reads an image from the given path. The image can be in NIfTI (.nii, .nii.gz) or DICOM format.
    If the path is a directory, it is assumed to be a DICOM series.
    
    # Parameters:
    - `path`: Path to the image.
    
    # Returns:
    - `MedImage`: The image.
"""
function read_image(path::AbstractString)
    p = lowercase(path)
    if isdir(path) || endswith(p, ".dcm")
        return _read_dicom(path)
    elseif endswith(p, ".nii") || endswith(p, ".nii.gz")
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

"""
    _read_dicom(path)
    
    Reads a DICOM image from the given path. 
    
    # Parameters:
    - `path`: Path to the DICOM image.
    
    # Returns:
    - `MedImage`: The DICOM image.
"""
function _read_dicom(path)
    dcms = isdir(path) ? DICOM.dcmdir_parse(path) : [DICOM.dcm_parse(path)]
    filter!(d -> haskey(d, (0x7fe0, 0x0010)), dcms)
    isempty(dcms) && error("No DICOM images found in $path")
    length(unique(d[(0x0020, 0x000e)] for d in dcms)) == 1 ||
        error("Folder contains more than one DICOM series: $path")

    d1 = dcms[1]
    ps, st = get_pixel_measures(d1)
    ps === nothing && error("PixelSpacing not found in $path")
    rows = Int(_scalar(d1, (0x0028, 0x0010), 0))
    cols = Int(_scalar(d1, (0x0028, 0x0011), 0))
    nf   = Int(_scalar(d1, (0x0028, 0x0008), 1))

    length(dcms) > 1 && nf > 1 &&
        error("Multiple multiframe DICOM files not supported: $path")

    if length(dcms) == 1 && nf > 1
        # multiframe: frames assumed already ordered along z
        frames = extract_multiframe_pixel_slices(d1, rows, cols, nf)
        slices = map(enumerate(frames)) do (i, f)
            slope, inter = get_rescale(d1; frame=i)
            f .* Float32(slope) .+ Float32(inter)
        end
        st === nothing && @warn "SliceThickness not found, z spacing = 1.0" path
        dz = something(st, 1.0)
        origin = zeros(3)
    else
        sort!(dcms; by = slice_pos)
        slices = map(dcms) do d
            px = d[(0x7fe0, 0x0010)]
            size(px) == (rows, cols) || (px = permutedims(px))
            size(px) == (rows, cols) || error("Unexpected pixel data size in $path")
            slope, inter = get_rescale(d)
            Float32.(px) .* Float32(slope) .+ Float32(inter)
        end
        has_pos = haskey(dcms[1], (0x0020, 0x0032)) && haskey(dcms[1], (0x0020, 0x0037))
        if has_pos && length(dcms) > 1
            z = slice_pos.(dcms)
            dz = abs(sum(diff(z)) / (length(z) - 1))
            maximum(abs.(abs.(diff(z)) .- dz)) <= 0.01 * dz ||
                @warn "Non-uniform slice spacing in $path" dz
        else
            dz = something(st, 1.0)
        end
        # dcms[1] is now the first slice in space (after sort!)
        origin = has_pos ? Float64.(collect(dcms[1][(0x0020, 0x0032)])) : zeros(3)
    end

    vol  = cat(slices...; dims = 3)          # (rows, cols, slices)
    data = permutedims(vol, (2, 1, 3))       # (cols, rows, slices), come il NIfTI
    return MedImage(data, [ps[2], ps[1], dz], origin, Matrix(Diagonal(ones(3))))
end