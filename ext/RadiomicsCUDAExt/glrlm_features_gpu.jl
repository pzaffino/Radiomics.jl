"""
    get_glrlm_features(img::AbstractArray{Float64},
                       mask::BitArray,
                       voxel_spacing::Vector{Float64};
                       n_bins::Union{Int,Nothing}=nothing,
                       bin_width::Union{Float64,Nothing}=nothing,
                       weighting_norm::Union{String,Nothing}=nothing,
                       get_raw_matrices::Bool=false,
                       features_std::Bool=false,
                       verbose::Bool=false,
                       gpu_data::GPUData)

    # Arguments
    - `img`: Input image.
    - `mask`: Binary ROI mask.
    - `voxel_spacing`: Voxel spacing.
    - `n_bins`: Number of gray level bins.
    - `bin_width`: Width of the gray level bins.
    - `weighting_norm`: Weighting norm used for GLRLM calculation.
    - `get_raw_matrices`: Flag used to return the raw GLRLM matrices.
    - `features_std`: Flag used to calculate the standard deviation of the GLRLM features.
    - `verbose`: Flag used to print progress information.
    - `gpu_data`: GPU data container

    # Returns
    GLRLM features
"""
function get_glrlm_features(img::AbstractArray{Float64},
    mask::BitArray,
    voxel_spacing::Vector{Float64};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    get_raw_matrices::Bool=false,
    features_std::Bool=false,
    verbose::Bool=false,
    gpu_data::GPUData)

    P_glrlm, max_run = compute_glrlm_gpu(
        gpu_data.texture_data.discretized_image,
        gpu_data.mask,
        gpu_data.mask_indices,
        gpu_data.texture_data.gl_lut,
        gpu_data.texture_data.num_gl,
        gpu_data.texture_data.min_gl
    )

    P_glrlm, _ = Radiomics.calculate_glrlm_matrix(Array{Int64}(undef, size(gpu_data.texture_data.discretized_image)),
        mask,
        voxel_spacing,
        weighting_norm,
        verbose,
        P_glrlm,
        gpu_data.texture_data.gray_levels_cpu,
        max_run
    )

    return Radiomics.get_glrlm_features(
        img,
        mask,
        voxel_spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        get_raw_matrices=get_raw_matrices,
        features_std=features_std,
        verbose=verbose,
        P_glrlm=P_glrlm,
        gray_levels=gpu_data.texture_data.gray_levels_cpu
    )

end

"""
    compute_glrlm_gpu(
        mask::CuArray{Bool},
        mask_indices::CuArray{Int},
        discretized_img::CuArray{Int},
        gray_levels::CuArray{Int},
        gl_lut::CuArray{Int},
        num_gl::Int,
        max_gl::Int,
        min_gl::Int)::Array{Float64}

    Computes the Gray Level Run Length Matrix (GLRLM) on the GPU.

    # Arguments
    - `mask`: ROI mask stored on the GPU.
    - `mask_indices`: Linear indices of valid ROI voxels.
    - `discretized_img`: Discretized image stored on the GPU.
    - `gray_levels`: Array containing all gray levels
    - `gl_lut`: Gray level look up table
    - `num_gl`: Number of gray levels 
    - `max_gl`: Maximum gray level 
    - `min_gl`: Minimum gray level

    # Returns
    - `Array{Float64}` containing the GLRLM 
    - `Int` actual max run
"""
function compute_glrlm_gpu(discretized_img::CuArray{Int},
    mask::CuArray{Bool},
    mask_indices::CuArray{Int},
    gl_lut::CuArray{Int},
    num_gl::Int,
    min_gl::Int)::Tuple{Array{Float64},Int}
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

    Nx, Ny = size(discretized_img)
    Nz = (dim == 3) ? size(discretized_img, 3) : 1
    num_indices = length(mask_indices)

    max_run_length_possible = maximum(size(discretized_img))

    num_angles = length(angles_x)

    P_glrlm = CUDA.zeros(Float64, num_gl, max_run_length_possible, num_angles)

    actual_max_run = CUDA.ones(Int, 1)


    blocks = (cld(num_indices, CUDA_BLOCK_WIDTH_2D), cld(num_angles, CUDA_BLOCK_HEIGHT_2D))
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks=blocks glrlm_kernel!(discretized_img, mask, mask_indices, gl_lut, P_glrlm, actual_max_run, Nx, Ny, Nz, angles_x, angles_y, angles_z, num_angles, num_indices, num_gl, min_gl, max_run_length_possible)
    actual_max = Array(actual_max_run)[1]
    return Array(P_glrlm[:, 1:actual_max, :]), actual_max
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

    x, y, z = decode_xyz(mask_indices[i], Nx, Ny, Nz)

    prev_x = x - dx
    prev_y = y - dy
    prev_z = z - dz

    if prev_x >= 1 && prev_x <= Nx && prev_y >= 1 && prev_y <= Ny && (Nz == 1 || (prev_z >= 1 && prev_z <= Nz))
        prev_idx = prev_x + (prev_y - 1) * Nx + (prev_z - 1) * Nx * Ny
        if mask[prev_idx] && img[prev_idx] == gl
            return
        end
    end

    run_length = 1

    next_x = x + dx
    next_y = y + dy
    next_z = z + dz

    while next_x >= 1 && next_x <= Nx && next_y >= 1 && next_y <= Ny && (Nz == 1 || (next_z >= 1 && next_z <= Nz))
        next_idx = next_x + (next_y - 1) * Nx + (next_z - 1) * Nx * Ny
        if !(mask[next_idx] && img[next_idx] == gl)
            break
        end

        run_length += 1

        next_x += dx
        next_y += dy
        next_z += dz
    end

    if run_length <= max_run_length
        bin = gl_idx + (run_length - 1) * num_gl + (j - 1) * num_gl * max_run_length

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