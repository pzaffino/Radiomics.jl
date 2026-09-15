""" 
    IMPORTANT: 
    - better information about how these kernels are called and executed can be found in the documentation of their respective caller functions
    - all CUDA.jl kernels must return `nothing`
"""

"""
    findall_kernel!(mask::CuDeviceArray{Bool},
                    idx::CuDeviceArray{Int},
                    valid_idx::CuDeviceArray{Int32},
                    mask_length::Int)
    
    Extracts all valid ROI indices 

    # Arguments:
    - `mask::CuArray`: The binary mask defining the region of interest stored on the GPU
    - `idx::CuArray`: The vector containing the position where each thread will write if the mask is true
    - `valid_idx::CuArray`: The vector where all valid ROI indices are stored
    - `mask_length::Int`: The length of `mask`

    # Caller functions:
    - `init_gpu` in `utils/utils_gpu/utils.jl`
"""
function findall_kernel!(mask::CuDeviceArray{Bool},
    idx::CuDeviceArray{Int},
    valid_idx::CuDeviceArray{Int32},
    mask_length::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > mask_length
        return nothing
    end

    if mask[i]
        valid_idx[idx[i]] = i
    end

    return nothing
end

"""
    assign_uniques!(img::CuDeviceArray{T},
                    is_boundary::CuDeviceArray{Int32},
                    idx::CuDeviceArray{Int},
                    uniques::CuDeviceArray{Int})

    Extracts all unique values inside an array 

    # Arguments
    - `img::CuDeviceArray`: The input image stored on the GPU 
    - `is_boundary::CuDeviceArray`: The binary array where each element indicates whether the corrisponding position is a boundary (1) or not (0)
    - `idx::CuDeviceArray`: The array where each element indicates the position where every thread will write if the corrisponding position is a boundary 
    - `uniques::CuDeviceArray`: The array where unique values will be stored 

    # Caller functions:
    - `unique_gpu` in `utils/utils_gpu/utils.jl`
"""
function assign_uniques!(img::CuDeviceArray{T},
    is_boundary::CuDeviceArray{Int32},
    counter::CuDeviceArray{Int},
    uniques::CuDeviceArray{T}) where T

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(is_boundary)
        return nothing
    end

    if is_boundary[i] != 0
        pos = CUDA.atomic_add!(pointer(counter), Int(1)) + 1
        uniques[pos] = img[i]
    end
    return nothing
end

"""
    set_boundaries!(xx::CuDeviceArray{T},
                    is_boundary::CuDeviceArray{Int32})

    Finds boundaries inside a sorted array. Example:
    x = [1, 1, 1, 2, 3, 3, 6, 6, 7]
    elements in position 1, 4, 5, 7, 8 are boundaries 

    # Arguments
    - `x`: Input array
    - `is_boundary`: A binary array containing boundary flags for the corresponding position

    # Caller functions:
    - `unique_gpu.jl` in `utils/utils_gpu/utils.jl`
"""
function set_boundaries!(x::CuDeviceArray{T},
    is_boundary::CuDeviceArray{Int32},
    num_of_uniques::CuDeviceArray{Int}) where T

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(x)
        return nothing
    end

    if i == 1
        CUDA.@atomic num_of_uniques[1] += 1
        is_boundary[i] = 1
    elseif x[i] != x[i-1]
        is_boundary[i] = true
        CUDA.@atomic num_of_uniques[1] += 1
    end

    return nothing
end

"""
    assign!(img::CuDeviceArray{Int},
            mask_indices::CuDeviceArray{Int},
            roi::CuDeviceArray{Int},
            n::Int)

    Extracts the intensity of all voxels belonging to the ROI 

    # Arguments
    - `img::CuDeviceArray`: The input image stored on the GPU
    - `mask_indices::CuDeviceArray`: The array containing all valid ROI indices
    - `roi::CuArray`: The array where the intensity of the voxels belonging to the ROI are stored
    - `n::Int`: The length of `mask_indices`

    # Caller functions:
    - `apply_mask` in `utils/utils_gpu/utils.jl`
"""
function assign!(img::CuDeviceArray{Int},
    mask_indices::CuDeviceArray{Int},
    roi::CuDeviceArray{Int},
    n::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n
        return nothing
    end

    roi[i] = img[mask_indices[i]]

    return nothing
end


"""
    bin_nbins_kernel!(img::CuDeviceArray{Float64}, 
                    mask_indices::CuDeviceArray{Int},
                    inv_bin_width::Float64, 
                    n_bins::Int,
                    vmin::Float64, 
                    disc::CuDeviceArray{Int},
                    n_of_indices::Int)

    Each CUDA thread processes one voxel inside the ROI.


    # Arguments
    - `img::CuDeviceArray`: Input image stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of voxels inside the ROI.
    - `inv_bin_width::Float64`: Inverse of the bin width.
    - `n_bins::Int`: Number of gray-level bins.
    - `vmin::Float64`: Minimum image intensity.
    - `disc::CuDeviceArray`: Discretized image stored on the GPU.
    - `n_of_indices::Int`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The discretized image `disc` is modified directly on the GPU.
"""
function bin_nbins_kernel!(img::CuDeviceArray{Float64},
    mask_indices::CuDeviceArray{Int},
    max_gl::CuDeviceArray{Int,1},
    min_gl::CuDeviceArray{Int,1},
    inv_bin_width::Float64,
    n_bins::Int,
    vmin::Float64,
    disc::CuDeviceArray{Int},
    n_of_indices::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img[mask_indices[i]]
    b = CUDA.min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
    disc[mask_indices[i]] = b

    CUDA.atomic_max!(pointer(max_gl, 1), Int(b))
    CUDA.atomic_min!(pointer(min_gl, 1), Int(b))

    return nothing
end


"""
    bin_width_kernel!(img::CuDeviceArray, 
                    mask_indices::CuDeviceArray{Int},
                    inv_bin_width::Float64, 
                    bin_offset::Int,
                    disc::CuDeviceArray{Int}, 
                    n_of_indices::Int)

    Each CUDA thread processes one voxel inside the ROI.

    # Arguments
    - `img::CuDeviceArray`: Input image stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of voxels inside the ROI.
    - `inv_bin_width::Float64`: Inverse of the bin width.
    - `bin_offset::Int`: Offset.
    - `disc::CuDeviceArray`: Discretized image stored on the GPU.
    - `n_of_indices::Int`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The discretized image `disc` is modified directly on the GPU.
"""
function bin_width_kernel!(img::CuDeviceArray{Float64},
    mask_indices::CuDeviceArray{Int},
    max_gl::CuDeviceArray{Int,1},
    min_gl::CuDeviceArray{Int,1},
    inv_bin_width::Float64, bin_offset::Int,
    disc::CuDeviceArray{Int},
    n_of_indices::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img[mask_indices[i]]
    b = Int(floor(v * inv_bin_width)) - bin_offset + 1
    disc[mask_indices[i]] = b

    CUDA.atomic_max!(pointer(max_gl, 1), Int(b))
    CUDA.atomic_min!(pointer(min_gl, 1), Int(b))

    return nothing
end


"""
    lut_kernel!(gray_levels::CuDeviceArray{Int}, 
                lut::CuDeviceArray{Int}, 
                min_gl::Int, 
                Ng::Int)

    CUDA kernel for constructing a gray level look up table (LUT).

    Each CUDA thread processes one gray level and assigns its corresponding
    index in the LUT

    # Arguments
    - `gray_levels`: Sorted array of unique gray levels present in the ROI
    - `lut`: Lookup table stored on the GPU.
    - `min_gl`: Minimum gray level value.
    - `Ng`: Number of gray levels.

    # Returns
    Returns `nothing`. The LUT is modified directly on the GPU.
"""
function lut_kernel!(gray_levels::CuDeviceArray{Int},
    lut::CuDeviceArray{Int},
    min_gl::Int,
    Ng::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > Ng
        return nothing
    end

    lut[Int(gray_levels[i])-min_gl+1] = i
    return nothing
end


"""
    mapped_disc_kernel!(disc::CuDeviceArray{Int}, 
                        mapped_disc::CuDeviceArray{Int},
                        mask::CuDeviceArray{Bool}, 
                        N::Int,
                        lut::CuDeviceArray{Int}, 
                        min_gl::Int)

    CUDA kernel for mapping discretized image gray levels to compact indices.

    Each CUDA thread processes one voxel.

    # Arguments
    - `disc::CuDeviceArray`: Discretized image.
    - `mapped_disc::CuDeviceArray`: Output array containing mapped indices.
    - `mask::CuDeviceArray`: ROI mask indicating valid voxels.
    - `N::Int`: Total number of voxels.
    - `lut::CuDeviceArray`: Gray level lookup table.
    - `min_gl::Int`: Minimum gray level value.

    # Returns
    Returns `nothing`. The mapped discretized image is modified directly on the GPU
"""
function mapped_disc_kernel!(disc::CuDeviceArray{Int},
    mapped_disc::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    N::Int,
    lut::CuDeviceArray{Int},
    min_gl::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > N
        return nothing
    end

    @inbounds if mask[i]
        mapped_disc[i] = lut[disc[i]-min_gl+1]
    end

    return nothing
end

"""
    classify_mask_indices!(mask_indices::CuDeviceArray{Int},
                          is_interior::CuDeviceArray{Bool},
                          is_border::CuDeviceArray{Bool},
                          Nx::Int,
                          Ny::Int,
                          Nz::Int,
                          num_indices::Int)

    Classifies ROI voxels into interior and border voxels.

    Each thread processes one voxel index from `mask_indices`

    # Arguments
    - `mask_indices`: Linear indices of voxels belonging to the ROI.
    - `is_interior`: Binary output array indicating interior voxels.
    - `is_border`: Binary output array indicating border voxels.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.
    - `num_indices`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The classification arrays are modified directly on the GPU.
"""
function classify_mask_indices!(
    mask_indices::CuDeviceArray{Int},
    is_interior::CuDeviceArray{Bool},
    is_border::CuDeviceArray{Bool},
    interior_length::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_indices::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_indices
        return nothing
    end

    x, y, z = decode_xyz(mask_indices[i], Nx, Ny, Nz)

    interior = false

    if Nz <= 1
        if (1 < x < Nx) && (1 < y < Ny)
            CUDA.@atomic interior_length[1] += 1
            interior = true
        end
    else
        if (1 < x < Nx) && (1 < y < Ny) && (1 < z < Nz)
            CUDA.@atomic interior_length[1] += 1
            interior = true
        end
    end

    is_interior[i] = interior
    is_border[i] = !interior

    return nothing
end

"""
    assign_border_interior!(mask_indices::CuDeviceArray{Int},
                            interior_mask::CuDeviceArray{Int},
                            border_mask::CuDeviceArray{Int},
                            interior_idx::CuDeviceArray{Int},
                            border_idx::CuDeviceArray{Int},
                            is_interior::CuDeviceArray{Bool},
                            is_border::CuDeviceArray{Bool},
                            num_indices::Int)

    Separates ROI voxel indices into interior and border lists.

    Each thread writes a voxel index into either the interior or border output
    array

    # Arguments
    - `mask_indices`: ROI voxel indices.
    - `interior_mask`: Output array containing interior voxel indices.
    - `border_mask`: Output array containing border voxel indices.
    - `interior_idx`: Write positions for interior voxels.
    - `border_idx`: Write positions for border voxels.
    - `is_interior`: Interior classification flags.
    - `is_border`: Border classification flags.
    - `num_indices`: Number of ROI voxels.

    # Returns
    Returns `nothing`. Output arrays are modified directly on the GPU.
"""
function assign_border_interior!(mask_indices::CuDeviceArray{Int},
    interior_mask::CuDeviceArray{Int},
    border_mask::CuDeviceArray{Int},
    interior_counter::CuDeviceArray{Int,1},
    border_counter::CuDeviceArray{Int,1},
    is_interior::CuDeviceArray{Bool},
    is_border::CuDeviceArray{Bool},
    num_indices::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_indices
        return nothing
    end

    if is_interior[i]
        pos = CUDA.@atomic interior_counter[1] += 1
        interior_mask[pos] = mask_indices[i]
    else
        pos = CUDA.@atomic border_counter[1] += 1
        border_mask[pos] = mask_indices[i]
    end

    return nothing
end

"""
    decode_xyz(idx::Int, Nx::Int, Ny::Int, Nz::Int)

    Converts a linear voxel index into 3D coordinates.

    # Arguments
    - `idx`: Linear voxel index.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.

    # Returns
    Returns `(x, y, z)` coordinates corresponding to the voxel position.
"""
@inline function decode_xyz(idx::Int, Nx::Int, Ny::Int, Nz::Int)::Tuple{Int,Int,Int}
    z = 1
    r = idx - 1
    if Nz > 1
        z = fld(r, Nx * Ny) + 1
        r = r % (Nx * Ny)
    end
    y = fld(r, Nx) + 1
    x = (r % Nx) + 1
    return x, y, z
end

"""
    encode_xyz(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)

    Converts 3D voxel coordinates into a linear index.

    # Arguments
    - `x`: X coordinate.
    - `y`: Y coordinate.
    - `z`: Z coordinate.
    - `Nx`: Image width.
    - `Ny`: Image height.

    # Returns
    Returns the linear index corresponding to `(x,y,z)`.
"""
@inline function encode_xyz(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)::Int
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function assign_uniques_partial!(uniques::CuDeviceArray{Int}, values::CuDeviceArray{Bool}, num_values::Int)
    return nothing
end

function assign_uniques_full!(
    values::CuDeviceArray{UInt32},
    uniques::CuDeviceArray{Int},
    counter::CuDeviceArray{Int},
    max_gl::Int,
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > max_gl
        return nothing
    end

    if values[i] != UInt32(0)
        pos = CUDA.atomic_add!(pointer(counter), 1)
        uniques[pos+1] = i
    end

    return nothing
end


function mark_existing_values!(
    img::CuDeviceArray{Int64},
    values::CuDeviceArray{UInt32},
    num_values::CuDeviceArray{Int64},
    img_size::Int64,
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > img_size
        return
    end

    value = img[i]

    old = CUDA.atomic_xchg!(
        pointer(values, value),
        UInt32(1),
    )

    if old == UInt32(0)
        CUDA.@atomic num_values[1] += Int64(1)
    end

    return
end
