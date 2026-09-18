"""
    get_glcm_features(img::AbstractArray{Float64},
                      mask::BitArray,
                      voxel_spacing::Vector{Float64};
                      n_bins::Union{Int,Nothing}=nothing,
                      bin_width::Union{Float64,Nothing}=nothing,
                      weighting_norm::Union{String,Nothing}=nothing,
                      get_raw_matrices::Bool=false,
                      features_std::Bool=false,
                      verbose::Bool=false,
                      gpu_data::GPUData)

    Compute GLCM features using GPU-accelerated GLCM calculation.

    # Arguments
    - `img`: Input image.
    - `mask`: Binary ROI mask.
    - `voxel_spacing`: Voxel spacing.
    - `n_bins`: Number of gray level bins.
    - `bin_width`: Width of the gray level bins.
    - `weighting_norm`: Weighting norm used for GLCM calculation.
    - `get_raw_matrices`: Flag used to return the raw GLCM matrices.
    - `features_std`: Flag used to calculate the standard deviation of the GLCM features.
    - `verbose`: Flag used to print progress information.
    - `gpu_data`: GPU data container

    # Returns
    GLCM features
"""
function get_glcm_features(img::AbstractArray{Float64},
    mask::BitArray,
    voxel_spacing::Vector{Float64};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    get_raw_matrices::Bool=false,
    features_std::Bool=false,
    verbose::Bool=false,
    gpu_data::GPUData)

    G_all = compute_glcm_gpu(
        gpu_data.texture_data.discretized_image,
        gpu_data
    )

    glcm_matrices, _ = Radiomics.calculate_glcm(img,
        mask,
        voxel_spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        verbose=verbose,
        G_all=G_all,
        gray_levels=gpu_data.texture_data.gray_levels_cpu,
        bin_width_used=gpu_data.texture_data.bin_width,
        n_bins_actual=gpu_data.texture_data.n_bins)

    return Radiomics.get_glcm_features(
        img,
        mask,
        voxel_spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        get_raw_matrices=get_raw_matrices,
        features_std=features_std,
        verbose=verbose,
        glcm_matrices=glcm_matrices,
        gray_levels=gpu_data.texture_data.gray_levels_cpu
    )
end


"""
    compute_glcm_gpu(disc::CuArray{Int}, 
                    gpu_data::GPUData)::Array{Float64}

    Compute the Gray Level Co-occurrence Matrix (GLCM) on the GPU.

    # Arguments
    - `disc`: Discretized image stored on the GPU.
    - `gpu_data`: GPU data container containing:
        - `gpu_data.img`: Original image stored on the GPU.
        - `gpu_data.mask`: ROI mask stored on the GPU.
        - `gpu_data.mask_indices`: Linear indices of valid ROI voxels.

    # Returns
    - `G_d`: Symmetric GLCM matrices on the CPU.

    # Notes
    - GLCM accumulation is performed on the GPU using atomic operations.
    - Symmetrization is performed on the CPU after GPU computation.
"""
function compute_glcm_gpu(disc::CuArray{Int},
    gpu_data::GPUData)::Array{Float64}
    dim = ndims(disc)

    if dim == 2
        dirs_x = CuArray([1, 0, 1, 1])
        dirs_y = CuArray([0, 1, 1, -1])
        dirs_z = CuArray([0, 0, 0, 0])
    else
        dirs_x = CuArray([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, -1])
        dirs_y = CuArray([0, 1, 0, 1, -1, 0, 0, 1, 1, 1, 1, -1, 1])
        dirs_z = CuArray([0, 0, 1, 0, 0, 1, -1, 1, -1, 1, -1, 1, 1])
    end
    max_gl = gpu_data.texture_data.max_gl
    min_gl = gpu_data.texture_data.min_gl
    Ng = gpu_data.texture_data.num_gl
    lut = gpu_data.texture_data.gl_lut

    mapped_disc = CUDA.zeros(Int, size(disc))
    Nx, Ny = size(mapped_disc)
    Nz = (dim == 3) ? size(mapped_disc, 3) : 1
    @cuda threads = CUDA_THREADS blocks = cld(length(disc), CUDA_THREADS) mapped_disc_kernel!(disc, mapped_disc, gpu_data.mask, length(disc), lut, min_gl)

    G_d = CUDA.zeros(Float64, length(dirs_x), Ng, Ng)

    n = length(gpu_data.mask_indices)
    num_dirs = length(dirs_x)

    blocks = (cld(n, CUDA_BLOCK_WIDTH_2D), cld(num_dirs, CUDA_BLOCK_HEIGHT_2D))
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = blocks glcm_kernel!(G_d, gpu_data.mask, gpu_data.mask_indices, mapped_disc, dirs_x, dirs_y, dirs_z, length(dirs_x), Nx, Ny, Nz, n)
    G_d = Array(G_d)
    for d in axes(G_d, 1)
        G_d[d, :, :] .+= G_d[d, :, :]'
    end
    return Array(G_d)
end

"""
    glcm_kernel!(G::CuDeviceArray{Float64}, 
                mask::CuDeviceArray{ool}, 
                mask_indices::CuDeviceArray{Int},
                mapped_disc::CuDeviceArray{Int}, 
                dirs_x::CuDeviceArray{Int}, 
                dirs_y::CuDeviceArray{Int},
                dirs_z::CuDeviceArray{Int}, 
                dirs_length::Int, 
                Nx::Int, 
                Ny::Int, 
                Nz::Int,
                num_valid::Int)

    CUDA kernel for computing the GLCM matrix.

    Each CUDA thread processes one voxel/direction pair.

    Symmetrization of the GLCM is performed on the CPU.

    # Arguments
    - `G::CuDeviceArray`: Output GLCM matrix stored on the GPU.
    - `mask::CuDeviceArray`: Binary ROI mask stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of valid voxels inside the ROI.
    - `mapped_disc::CuDeviceArray`: Discretized image.
    - `dirs_x::CuDeviceArray`: x components of the directions.
    - `dirs_y::CuDeviceArray`: y components of the directions.
    - `dirs_z::CuDeviceArray`: z components of the directions.
    - `dirs_length::Int`: Number of directions.
    - `Nx::Int`: Image width
    - `Ny::Int`: Image height
    - `Nz::Int`: Image depth
    - `num_valid::Int`: Number of valid voxels in the ROI

    # Returns
    Returns `nothing`. The GLCM matrix `G` is modified directly on the GPU
"""
function glcm_kernel!(G::CuDeviceArray{Float64},
    mask::CuDeviceArray{Bool},
    mask_indices::CuDeviceArray{Int},
    mapped_disc::CuDeviceArray{Int},
    dirs_x::CuDeviceArray{Int},
    dirs_y::CuDeviceArray{Int},
    dirs_z::CuDeviceArray{Int},
    dirs_length::Int,
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_valid::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x # maps threads to mask
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y # maps threads to directions

    if i > num_valid || j > dirs_length
        return nothing
    end

    x, y, z = decode_xyz(mask_indices[i], Nx, Ny, Nz)

    dx = dirs_x[j]
    dy = dirs_y[j]
    dz = Nz > 1 ? dirs_z[j] : 0

    nx = x + dx
    ny = y + dy
    nz = z + dz

    if nx < 1 || nx > Nx || ny < 1 || ny > Ny
        return
    end

    if Nz > 1
        if nz < 1 || nz > Nz
            return nothing
        end
    end

    if !mask[nx, ny, nz]
        return nothing
    end

    i_disc = mapped_disc[x, y, z]
    j_disc = mapped_disc[nx, ny, nz]

    # only perform one sum, symmetrization is applied on the CPU because it's faster this way -> fewer threads wait for synchronization due to race condition
    CUDA.@atomic G[j, i_disc, j_disc] += 1.0

    return nothing
end