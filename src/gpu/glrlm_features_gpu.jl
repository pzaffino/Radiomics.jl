"""
    compute_glrlm_gpu(
        mask::CuArray{Bool},
        mask_indices::CuArray{Int},
        discretized_img::CuArray{Int}
    )::Array{Float64}

    Computes the Gray Level Run Length Matrix (GLRLM) on the GPU.

    # Arguments
    - `mask`: ROI mask stored on the GPU.
    - `mask_indices`: Linear indices of valid ROI voxels.
    - `discretized_img`: Discretized image stored on the GPU.

    # Returns
    - `Array{Float64}` containing the GLRLM 
"""
function compute_glrlm_gpu(mask::CuArray{Bool}, mask_indices::CuArray{Int}, discretized_img::CuArray{Int})::Array{Float64}
    dim = ndims(discretized_img)

    if dim == 2
        angles_x = CuArray([1, 0, 1, 1, -1, 0, -1, -1])
        angles_y = CuArray([0, 1, 1, -1, 0, -1, -1, 1])
        angles_z = CuArray([0, 0, 0, 0, 0, 0, 0, 0])
    else
        angles_x = CuArray([1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1])
        angles_y = CuArray([0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1])
        angles_z = CuArray([0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    end

    masked_img = apply_mask(discretized_img, mask_indices)
    gray_levels = unique_gpu(masked_img)
    num_gl = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
    gl_lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    Nx, Ny = size(discretized_img)
    Nz = (dim == 3) ? size(discretized_img, 3) : 1
    num_indices = length(mask_indices)

    @cuda threads = CUDA_THREADS blocks = cld(num_gl, CUDA_THREADS) lut_kernel!(gray_levels, gl_lut, min_gl, num_gl)

    max_run_length_possible = maximum(size(discretized_img))

    num_angles = length(angles_x)

    P_glrlm = CUDA.zeros(Float64, num_gl, max_run_length_possible, num_angles)

    actual_max_run = CUDA.ones(Int, 1)

    blocks_x = cld(num_indices, CUDA_BLOCK_WIDTH_2D)
    blocks_y = cld(num_angles, CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (blocks_x, blocks_y) glrlm_kernel!(discretized_img, mask, mask_indices, gl_lut, P_glrlm, actual_max_run, Nx, Ny, Nz, angles_x, angles_y, angles_z, num_angles, num_indices, num_gl, min_gl, max_run_length_possible)
    CUDA.synchronize()

    actual_max = Array(actual_max_run)[1]
    return Array(P_glrlm[:, 1:actual_max, :])
end

"""
    glrlm_kernel!(img::CuArray{Int}, 
                mask::CuArray{Bool}, 
                mask_indices::CuArray{Int},
                gl_lut::CuArray{Int}, 
                P_glrlm::CuArray{Int}, 
                actual_max_run::CuArray{Int},
                Nx::Int, 
                Ny::Int, 
                Nz::Int,
                angles_x::CuArray{Int}, 
                angles_y::CuArray{Int}, 
                angles_z::CuArray{Int},
                num_angles::Int, 
                num_indices::Int,
                num_gl::Int, 
                min_gl::Int, 
                max_run_length::Int
    )

    CUDA kernel for computing the Gray Level Run Length Matrix (GLRLM).

    Each CUDA thread processes one voxel/direction pair.

    # Arguments
    - `img`: Discretized image stored on the GPU.
    - `mask`: Binary ROI mask stored on the GPU.
    - `mask_indices`: Indices of valid voxels inside the ROI.
    - `gl_lut`: Lookup table mapping gray levels to GLRLM indices.
    - `P_glrlm`: Output GLRLM matrix stored on the GPU.
    - `actual_max_run`: Single element array storing the maximum detected run length.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.
    - `angles_x`: x components of the run directions.
    - `angles_y`: y components of the run directions.
    - `angles_z`: z components of the run directions.
    - `num_angles`: Number of directions.
    - `num_indices`: Number of valid voxels in the ROI.
    - `num_gl`: Number of unique gray levels.
    - `min_gl`: Minimum gray level in the discretized image.
    - `max_run_length`: Maximum run length.

    # Returns
    Returns `nothing`. The GLRLM matrix `P_glrlm` and the maximum run length
    `actual_max_run` are modified directly on the GPU.
"""

function glrlm_kernel!(
    img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    mask_indices::CuDeviceArray{Int},
    gl_lut::CuDeviceArray{Int},
    P_glrlm::CuDeviceArray{Float64},
    actual_max_run::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    angles_x::CuDeviceArray{Int},
    angles_y::CuDeviceArray{Int},
    angles_z::CuDeviceArray{Int},
    num_angles::Int,
    num_indices::Int,
    num_gl::Int,
    min_gl::Int,
    max_run_length::Int
)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_indices || j > num_angles
        return
    end

    dx = angles_x[j]
    dy = angles_y[j]
    dz = angles_z[j]

    idx = mask_indices[i]

    gl = img[idx]
    gl_idx = gl_lut[gl-min_gl+1]

    z = 1
    r = idx - 1

    if Nz > 1
        z = fld(r, Nx * Ny) + 1
        r = r % (Nx * Ny)
    end

    y = fld(r, Nx) + 1
    x = (r % Nx) + 1

    prev_x = x - dx
    prev_y = y - dy
    prev_z = z - dz

    if prev_x >= 1 && prev_x <= Nx &&
       prev_y >= 1 && prev_y <= Ny &&
       (Nz == 1 || (prev_z >= 1 && prev_z <= Nz))

        prev_idx = prev_x +
                   (prev_y - 1) * Nx +
                   (prev_z - 1) * Nx * Ny

        if mask[prev_idx] && img[prev_idx] == gl
            return
        end
    end

    run_length = 1

    next_x = x + dx
    next_y = y + dy
    next_z = z + dz

    while next_x >= 1 && next_x <= Nx &&
              next_y >= 1 && next_y <= Ny &&
              (Nz == 1 || (next_z >= 1 && next_z <= Nz))

        next_idx = next_x +
                   (next_y - 1) * Nx +
                   (next_z - 1) * Nx * Ny

        if !(mask[next_idx] && img[next_idx] == gl)
            break
        end

        run_length += 1

        next_x += dx
        next_y += dy
        next_z += dz
    end

    if run_length <= max_run_length
        bin = gl_idx +
              (run_length - 1) * num_gl +
              (j - 1) * num_gl * max_run_length

        CUDA.atomic_add!(
            pointer(P_glrlm, bin),
            Float64(1)
        )
        CUDA.atomic_max!(
            pointer(actual_max_run, 1),
            Int(run_length)
        )
    end

    return
end