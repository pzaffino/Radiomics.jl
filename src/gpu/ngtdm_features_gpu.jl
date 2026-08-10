"""
    compute_ngtdm_gpu(
        discretized_img::CuArray{Int},
        mask::CuArray{Bool},
        mask_indices::CuArray{Int}
    )::Tuple{Array{Float64}, Array{Int}}

    Computes the Neighborhood Gray-Tone Difference Matrix (NGTDM) on the GPU.

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask`: Binary ROI mask stored on the GPU.
    - `mask_indices`: Linear indices of ROI voxels.

    # Returns
    - `P_ngtdm`: NGTDM matrix transferred back to the CPU
    - `gray_levels`: Array of gray levels present in the ROI transferred back to the CPU
"""
function compute_ngtdm_gpu(discretized_img::CuArray{Int},
    mask::CuArray{Bool},
    mask_indices::CuArray{Int})::Tuple{Array{Float64},Array{Int}}

    masked_img = apply_mask(discretized_img, mask_indices)

    gray_levels = unique_gpu(masked_img)
    num_gl = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
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
    is_interior = CUDA.zeros(Int, num_indices)
    is_border = CUDA.ones(Int, num_indices)

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) classify_mask_indices!(mask_indices, is_interior, is_border, Nx, Ny, Nz, num_indices)

    interior_mask = CUDA.zeros(Int, sum(is_interior))
    border_mask = CUDA.zeros(Int, sum(is_border))

    interior_idx = cumsum(is_interior)
    border_idx = cumsum(is_border)

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) assign_border_interior!(mask_indices, interior_mask, border_mask, interior_idx, border_idx, is_interior, is_border, num_indices)

    n_int = length(interior_mask)
    n_bord = length(border_mask)

    P_ngtdm = CUDA.zeros(Float64, num_gl, 3)

    shmem_size = num_gl * sizeof(Int) + num_gl * sizeof(Float64)

    @cuda threads = CUDA_THREADS blocks = cld(n_int, CUDA_THREADS) shmem = shmem_size ngtdm_neighborhood_count_interior!(discretized_img, mask, interior_mask, gl_lut, offsets_x, offsets_y, offsets_z, P_ngtdm, Nx, Ny, Nz, min_gl, num_gl, n_int, num_offsets)
    @cuda threads = CUDA_THREADS blocks = cld(n_bord, CUDA_THREADS) shmem = shmem_size ngtdm_neighborhood_count_border!(discretized_img, mask, border_mask, gl_lut, offsets_x, offsets_y, offsets_z, P_ngtdm, Nx, Ny, Nz, min_gl, num_gl, n_bord, num_offsets)

    return Array(P_ngtdm), Array(gray_levels)
end

"""
    ngtdm_neighborhood_count_interior!(
        discretized_img::CuDeviceArray{Int},
        mask::CuDeviceArray{Bool},
        interior_mask::CuDeviceArray{Int},
        gl_map::CuDeviceArray{Int},
        offsets_x::CuDeviceArray{Int},
        offsets_y::CuDeviceArray{Int},
        offsets_z::CuDeviceArray{Int},
        P_ngtdm::CuDeviceArray{Float64},
        Nx::Int,
        Ny::Int,
        Nz::Int,
        min_gl::Int,
        num_gl::Int,
        num_interior::Int,
        num_offsets::Int
    )

    Computes the NGTDM matrix  for interior ROI voxels.

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask`: Binary ROI mask.
    - `interior_mask`: List of interior ROI voxel indices.
    - `gl_map`: Gray-level lookup table.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `P_ngtdm`: Output NGTDM matrix.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `min_gl`: Minimum gray level.
    - `num_gl`: Number of gray levels.
    - `num_interior`: Number of interior ROI voxels.
    - `num_offsets`: Number of offsets.

    # Returns
    Returns `nothing`. The NGTDM matrix is updated directly on the GPU.
"""
function ngtdm_neighborhood_count_interior!(
    discretized_img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    interior_mask::CuDeviceArray{Int},
    gl_map::CuDeviceArray{Int},
    offsets_x::CuDeviceArray{Int},
    offsets_y::CuDeviceArray{Int},
    offsets_z::CuDeviceArray{Int},
    P_ngtdm::CuDeviceArray{Float64},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    min_gl::Int,
    num_gl::Int,
    num_interior::Int,
    num_offsets::Int,)

    sh_counts = @cuDynamicSharedMem(Int, num_gl)
    sh_sums = @cuDynamicSharedMem(Float64, num_gl, num_gl*sizeof(Int))

    tid = threadIdx().x
    i = tid + (blockIdx().x - 1) * blockDim().x

    if tid <= num_gl
        sh_counts[tid] = 0
        sh_sums[tid] = 0.0
    end

    CUDA.sync_threads()

    if i <= num_interior

        idx = interior_mask[i]
        x, y, z = decode_xyz(idx, Nx, Ny, Nz)

        gl = discretized_img[idx]
        gl_idx = gl_map[gl-min_gl+1]

        neighborhood_sum = 0
        neighborhood_count = 0

        for o in 1:num_offsets
            nx = x + offsets_x[o]
            ny = y + offsets_y[o]
            nz = z + offsets_z[o]
            nidx = encode_xyz(nx, ny, nz, Nx, Ny)

            if mask[nidx]
                neighborhood_sum += discretized_img[nidx]
                neighborhood_count += 1
            end
        end

        if neighborhood_count > 0
            neighborhood_avg = neighborhood_sum / neighborhood_count
            CUDA.@atomic sh_counts[gl_idx] += 1
            CUDA.@atomic sh_sums[gl_idx] += abs(gl - neighborhood_avg)
        end
    end

    CUDA.sync_threads()

    if tid <= num_gl
        if sh_counts[tid] > 0
            CUDA.@atomic P_ngtdm[tid, 1] += sh_counts[tid]
            CUDA.@atomic P_ngtdm[tid, 2] += sh_sums[tid]
            P_ngtdm[gl_idx, 3] = gl
        end
    end
    return nothing
end

"""
    ngtdm_neighborhood_count_border!(
        discretized_img::CuDeviceArray{Int},
        mask::CuDeviceArray{Bool},
        border_mask::CuDeviceArray{Int},
        gl_map::CuDeviceArray{Int},
        offsets_x::CuDeviceArray{Int},
        offsets_y::CuDeviceArray{Int},
        offsets_z::CuDeviceArray{Int},
        P_ngtdm::CuDeviceArray{Float64},
        Nx::Int,
        Ny::Int,
        Nz::Int,
        min_gl::Int,
        num_gl::Int,
        num_border::Int,
        num_offsets::Int
    )

    Computes the NGTDM matrix for border ROI voxels.

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask`: Binary ROI mask.
    - `border_mask`: List of border ROI voxel indices.
    - `gl_map`: Gray-level lookup table.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `P_ngtdm`: Output NGTDM matrix.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `min_gl`: Minimum gray level.
    - `num_gl`: Number of gray levels.
    - `num_border`: Number of border ROI voxels.
    - `num_offsets`: Number of offsets.

    # Returns
    Returns `nothing`. The NGTDM matrix is updated directly on the GPU.
"""
function ngtdm_neighborhood_count_border!(
    discretized_img::CuDeviceArray{Int}, mask::CuDeviceArray{Bool}, border_mask::CuDeviceArray{Int}, gl_map::CuDeviceArray{Int}, offsets_x::CuDeviceArray{Int}, offsets_y::CuDeviceArray{Int}, offsets_z::CuDeviceArray{Int}, P_ngtdm::CuDeviceArray{Float64}, Nx::Int, Ny::Int, Nz::Int, min_gl::Int, num_gl::Int, num_border::Int, num_offsets::Int,)

    sh_counts = @cuDynamicSharedMem(Int, num_gl)
    sh_sums = @cuDynamicSharedMem(Float64, num_gl, num_gl*sizeof(Int))

    tid = threadIdx().x
    i = tid + (blockIdx().x - 1) * blockDim().x

    if tid <= num_gl
        sh_counts[tid] = 0
        sh_sums[tid] = 0.0
    end

    CUDA.sync_threads()

    if i <= num_border
        idx = border_mask[i]

        x, y, z = decode_xyz(idx, Nx, Ny, Nz)

        gl = discretized_img[idx]
        gl_idx = gl_map[gl-min_gl+1]

        neighborhood_sum = 0
        neighborhood_count = 0

        for o in 1:num_offsets
            nx = x + offsets_x[o]
            ny = y + offsets_y[o]
            nz = z + offsets_z[o]

            if (1 <= nx <= Nx) && (1 <= ny <= Ny) && (1 <= nz <= Nz)
                nidx = encode_xyz(nx, ny, nz, Nx, Ny)
                if mask[nidx]
                    neighborhood_sum += discretized_img[nidx]
                    neighborhood_count += 1
                end
            end
        end

        if neighborhood_count > 0
            neighborhood_avg = neighborhood_sum / neighborhood_count
            CUDA.@atomic P_ngtdm[gl_idx, 1] += 1
            CUDA.@atomic P_ngtdm[gl_idx, 2] += abs(gl - neighborhood_avg)
            P_ngtdm[gl_idx, 3] = gl
        end
    end

    CUDA.sync_threads()

    if tid <= num_gl
        if sh_counts[tid] > 0
            CUDA.@atomic P_ngtdm[tid, 1] += sh_counts[tid]
            CUDA.@atomic P_ngtdm[tid, 2] += sh_sums[tid]
            P_ngtdm[gl_idx, 3] = gl
        end
    end
    return nothing
end