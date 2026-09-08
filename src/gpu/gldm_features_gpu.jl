"""
    compute_gldm_gpu(discretized_img::CuArray{Int},
        mask::CuArray{Bool},
        mask_indices::CuArray{Int},
        gray_levels::CuArray{Int},
        num_gl::Int,
        max_gl::Int,
        min_gl::Int,
        gldm_a::Int)::Tuple{CuArray{Int},CuArray{Int}}

    Computes the Gray Level Dependence Matrix (GLDM) on the GPU.

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask`: ROI mask stored on the GPU.
    - `mask_indices`: Linear indices of ROI voxels.
    - `gray_levels`: Array containing all gray levels
    - `num_gl`: Number of gray levels 
    - `max_gl`: Maximum gray level 
    - `min_gl`: Minimum gray level
    - `gldm_a`: Maximum gray-level difference allowed for dependence.

    # Returns
    - `Tuple{Matrix{Int}, Array{Int}}`:
        - `P_gldm`: GLDM matrix transferred to the CPU.
        - `gray_levels`: Unique gray levels present in the ROI transferred to the CPU.
"""
function compute_gldm_gpu(
    discretized_img::CuArray{Int},
    mask::CuArray{Bool},
    mask_indices::CuArray{Int},
    gray_levels::CuArray{Int},
    num_gl::Int,
    max_gl::Int,
    min_gl::Int,
    gldm_a::Int)::Tuple{Matrix{Int},Array{Int}}

    gl_lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    @cuda threads = CUDA_THREADS blocks = cld(num_gl, CUDA_THREADS) lut_kernel!(
        gray_levels, gl_lut, min_gl, num_gl)

    n_dims = ndims(discretized_img)
    sz = size(discretized_img)

    Nx, Ny = sz
    Nz = (n_dims == 3) ? sz[3] : 1

    if n_dims == 2
        offsets_x = CuArray([-1, -1, -1, 0, 0, 1, 1, 1])
        offsets_y = CuArray([-1, 0, 1, -1, 1, -1, 0, 1])
        offsets_z = CuArray([0, 0, 0, 0, 0, 0, 0, 0])
    else
        offsets_x = CuArray([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        offsets_y = CuArray([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1])
        offsets_z = CuArray([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
    end
    num_offsets = length(offsets_x)

    num_indices = length(mask_indices)
    is_interior = CUDA.zeros(Bool, num_indices)
    is_border = CUDA.ones(Bool, num_indices)

    interior_length = CuArray([0])

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) classify_mask_indices!(mask_indices, is_interior, is_border, interior_length, Nx, Ny, Nz, num_indices)

    interior_length = Array(interior_length)[1]
    interior_mask = CUDA.zeros(Int, interior_length)
    border_mask = CUDA.zeros(Int, num_indices - interior_length)

    interior_counter = CuArray([1])
    border_counter = CuArray([1])

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) assign_border_interior!(mask_indices, interior_mask, border_mask, interior_counter, border_counter, is_interior, is_border, num_indices)

    n_int = length(interior_mask)
    n_bord = length(border_mask)

    max_dependence = 3^n_dims
    P_gldm = CUDA.zeros(Int, num_gl, max_dependence)

    dep_interior = CUDA.ones(Int, n_int)
    dep_border = CUDA.ones(Int, n_bord)

    bx_int = cld(n_int, CUDA_BLOCK_WIDTH_2D)
    bx_bord = cld(n_bord, CUDA_BLOCK_WIDTH_2D)
    by = cld(num_offsets, CUDA_BLOCK_HEIGHT_2D)

    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (bx_int, by) gldm_interior_dependence!(discretized_img, mask, interior_mask, dep_interior, offsets_x, offsets_y, offsets_z, Nx, Ny, Nz, n_int, num_offsets, gldm_a)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (bx_bord, by) gldm_border_dependence!(discretized_img, mask, border_mask, dep_border, offsets_x, offsets_y, offsets_z, Nx, Ny, Nz, n_bord, num_offsets, gldm_a)

    @cuda threads = CUDA_THREADS blocks = cld(n_bord, CUDA_THREADS) gldm_kernel!(discretized_img, border_mask, gl_lut, dep_border, min_gl, P_gldm, n_bord)
    @cuda threads = CUDA_THREADS blocks = cld(n_int, CUDA_THREADS) gldm_kernel!(discretized_img, interior_mask, gl_lut, dep_interior, min_gl, P_gldm, n_int)

    col_has_data = vec(any(!iszero, P_gldm; dims=1))
    col_has_data_cpu = Array(col_has_data)
    last_col = findlast(col_has_data_cpu)
    last_col = last_col === nothing ? 0 : last_col
    P_gldm = P_gldm[:, 1:last_col]
    return Array(P_gldm), Array(gray_levels)
end

"""
    gldm_kernel!(discretized_img::CuDeviceArray{Int},
                            idx_list::CuDeviceArray{Int},
                            gl_lut::CuDeviceArray{Int},
                            dependence_count::CuDeviceArray{Int},
                            min_gl::Int,
                            P_gldm::CuDeviceArray{Int},
                            n::Int)

    Builds the GLDM histogram from voxel dependence counts.

    Each thread maps a voxel gray level and its dependence count into the
    corresponding GLDM histogram bin

    # Arguments
    - `discretized_img`: Discretized image.
    - `idx_list`: List of voxel indices.
    - `gl_lut`: Gray-level lookup table.
    - `dependence_count`: Computed dependence values.
    - `min_gl`: Minimum gray level.
    - `P_gldm`: Output GLDM matrix.
    - `n`: Number of voxels.

    # Returns
    Returns `nothing`. The GLDM matrix is updated directly on the GPU.
"""
function gldm_kernel!(
    discretized_img::CuDeviceArray{Int},
    idx_list::CuDeviceArray{Int},
    gl_lut::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    min_gl::Int,
    P_gldm::CuDeviceArray{Int},
    n::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > n
        return nothing
    end

    idx = idx_list[i]
    gl = discretized_img[idx]
    gl_idx = gl_lut[gl-min_gl+1]

    CUDA.@atomic P_gldm[gl_idx, dependence_count[i]] += 1

    return nothing
end

"""
    gldm_interior_dependence!(discretized_img::CuDeviceArray{Int},
                              mask::CuDeviceArray{Bool},
                              interior_mask::CuDeviceArray{Int},
                              dependence_count::CuDeviceArray{Int},
                              offsets_x::CuDeviceArray{Int},
                              offsets_y::CuDeviceArray{Int},
                              offsets_z::CuDeviceArray{Int},
                              Nx::Int,
                              Ny::Int,
                              Nz::Int,
                              num_interior::Int,
                              num_offsets::Int,
                              gldm_a::Int)

    Computes gray level dependence counts for interior ROI voxels.

    Each thread evaluates one voxel/offset pair

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask:`: Binary ROI mask.
    - `interior_mask`: Interior voxel indices.
    - `dependence_count`: Output dependence counts.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `num_interior`: Number of interior voxels.
    - `num_offsets`: Number of neighborhood offsets.
    - `gldm_a:`: Maximum allowed gray-level difference.

    # Returns
    Returns `nothing`. Dependence counts are updated directly on the GPU.
"""
function gldm_interior_dependence!(
    discretized_img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    interior_mask::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    offsets_x::CuDeviceArray{Int},
    offsets_y::CuDeviceArray{Int},
    offsets_z::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_interior::Int,
    num_offsets::Int,
    gldm_a::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    o = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_interior || o > num_offsets
        return nothing
    end

    idx = interior_mask[i]
    x, y, z = decode_xyz(idx, Nx, Ny, Nz)

    nx = x + offsets_x[o]
    ny = y + offsets_y[o]
    nz = z + offsets_z[o]
    nidx = encode_xyz(nx, ny, nz, Nx, Ny)

    gl = discretized_img[idx]
    if mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
        CUDA.@atomic dependence_count[i] += 1
    end
    return nothing
end

"""
    gldm_border_dependence!(discretized_img::CuDeviceArray{Int},
                              mask::CuDeviceArray{Bool},
                              border_mask::CuDeviceArray{Int},
                              dependence_count::CuDeviceArray{Int},
                              offsets_x::CuDeviceArray{Int},
                              offsets_y::CuDeviceArray{Int},
                              offsets_z::CuDeviceArray{Int},
                              Nx::Int,
                              Ny::Int,
                              Nz::Int,
                              num_border::Int,
                              num_offsets::Int,
                              gldm_a::Int)

    Computes gray level dependence counts for border ROI voxels.

    This kernel is equivalent to `gldm_interior_dependence!` 

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask:`: Binary ROI mask.
    - `border_mask`: Border voxel indices.
    - `dependence_count`: Output dependence counts.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `num_border`: Number of border voxels.
    - `num_offsets`: Number of neighborhood offsets.
    - `gldm_a:`: Maximum allowed gray-level difference.

    # Returns
    Returns `nothing`. Dependence counts are modified directly on the GPU.
"""
function gldm_border_dependence!(
    discretized_img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    border_mask::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    offsets_x::CuDeviceArray{Int},
    offsets_y::CuDeviceArray{Int},
    offsets_z::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_border::Int,
    num_offsets::Int,
    gldm_a::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    o = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_border || o > num_offsets
        return nothing
    end

    idx = border_mask[i]
    x, y, z = decode_xyz(idx, Nx, Ny, Nz)

    nx = x + offsets_x[o]
    ny = y + offsets_y[o]
    nz = z + offsets_z[o]

    if (1 <= nx <= Nx) && (1 <= ny <= Ny) && (1 <= nz <= Nz)
        nidx = encode_xyz(nx, ny, nz, Nx, Ny)
        gl = discretized_img[idx]
        if mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
            CUDA.@atomic dependence_count[i] += 1
        end
    end
    return nothing
end