using DICOM
using NRRD
using NIfTI

using FileIO
using LinearAlgebra: Diagonal

include("DICOM/utils_dicom.jl")

"""
    MedImage{T,N}
    
    A medical image with the given type T and number of dimensions N.
    
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
    
    Reads an image from the given path. The image can be in NIfTI (.nii, .nii.gz),
    NRRD (.nrrd, .nhdr) or DICOM format. If the path is a directory, it is assumed
    to be a DICOM series.

    # Arguments:
    - `path`: The path to the image.

    # Returns:
    - `MedImage`: The medical image.
"""
function read_image(path::AbstractString)
    p = lowercase(path)
    if isdir(path) || endswith(p, ".dcm")
        return _read_dicom(path)
    elseif endswith(p, ".nii") || endswith(p, ".nii.gz")
        return _read_nifti(path)
    elseif endswith(p, ".nrrd") || endswith(p, ".nhdr")
        return _read_nrrd(path)
    else
        error("Not supported format: $path")
    end
end

"""
    _resolve_inputs(img, mask, spacing, suv_ibsi_normalization)

    This function is used to resolve the inputs to the given function.
    It takes the image, the mask, the spacing and a boolean to indicate whether to normalize the SUV.

    # Arguments:
    - `img`: The image data or path to the image.
    - `mask`: The mask data or path to the mask.
    - `spacing`: The spacing of the voxels.
    - `suv_ibsi_normalization`: Whether to normalize the SUV.

    # Returns:
    - `img`: The image data.
    - `mask`: The mask data.
    - `spacing`: The spacing of the voxels.
"""
function _resolve_inputs(img, mask, spacing, suv_ibsi_normalization::Bool=false)
    img_is_path = img isa AbstractString
    mask_is_path = mask isa AbstractString

    img_is_path == mask_is_path ||
        error("Image and mask must be both paths or both arrays")

    if img_is_path
        m = read_image(mask)
        if suv_ibsi_normalization
            ndims(m.data) == 3 || error("suv_ibsi_normalization=true requires a 3D mask")
            roi_slices = [k for k in axes(m.data, 3) if any(!iszero, @view m.data[:, :, k])]
            i = _read_dicom_suv(img, roi_slices)
        else
            i = read_image(img)
        end
        all(isapprox.(i.spacing, m.spacing; atol=1e-3)) ||
            error("Different Spacing: image $(i.spacing), mask $(m.spacing)")
        return i.data, m.data, isnothing(spacing) ? i.spacing : spacing
    end

    suv_ibsi_normalization &&
        error("suv_ibsi_normalization=true is only supported with DICOM paths, not with arrays")
    isnothing(spacing) && error("With arrays, spacing is required")
    return img, mask, spacing
end

"""
    _read_nifti(path)
    
    Reads a NIfTI image from the given path.

    # Arguments:
    - `path`: The path to the image.

    # Returns:
    - `MedImage`: The medical image.
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
    
    Reads a DICOM image (series or single file) with rescale slope/intercept applied.

    # Arguments:
    - `path`: The path to the image.

    # Returns:
    - `MedImage`: The medical image.
"""
function _read_dicom(path)
    dcms, nf = _load_dicom_series(path)
    d1 = dcms[1]
    ps, st = get_pixel_measures(d1)
    ps === nothing && error("PixelSpacing not found in $path")
    rows = Int(_scalar(d1, (0x0028, 0x0010), 0))
    cols = Int(_scalar(d1, (0x0028, 0x0011), 0))

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
        origin = has_pos ? Float64.(collect(dcms[1][(0x0020, 0x0032)])) : zeros(3)
    end

    vol = cat(slices...; dims=3)          # (rows, cols, slices)
    data = permutedims(vol, (2, 1, 3))       # (cols, rows, slices), come il NIfTI
    return MedImage(data, [ps[2], ps[1], dz], origin, Matrix(Diagonal(ones(3))))
end

"""
    _read_nrrd(path)
    
    Reads an NRRD image from the given path.
    
    # Parameters:
    - `path`: Path to the NRRD image.
    
    # Returns:
    - `MedImage`: The NRRD image.
"""
function _read_nrrd(path)
    img = FileIO.load(path)
    data = Array(img.data)
    N = ndims(data)
    sp = [Float64(step(img.axes[i].val)) for i in 1:N]
    org = [Float64(first(img.axes[i].val)) for i in 1:N]
    all(>(0), sp) || error("Non-positive spacing in $path: $sp")

    hdr = open(path) do io
        lines = String[]
        for l in eachline(io)
            isempty(l) && break
            push!(lines, l)
        end
        lines
    end
    for l in hdr
        startswith(l, "space directions:") || continue
        vs = [parse.(Float64, split(m.captures[1], ',')) for m in eachmatch(r"\(([^)]*)\)", l)]
        length(vs) == N || continue
        M = reduce(hcat, vs)
        off = maximum(abs(M[i, j]) for i in 1:N, j in 1:N if i != j)
        off <= 1e-6 ||
            @warn "NRRD with non-diagonal space directions (oblique): direction ignored" path
    end

    return MedImage(data, sp, org, Matrix(Diagonal(ones(N))))
end