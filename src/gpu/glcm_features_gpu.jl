"""
    compute_glcm_gpu(disc::CuArray{Int}, 
                    gray_levels::CuArray{Int}, 
                    gpu_data::GPUData)::Array{Float64}

    Compute the Gray Level Co-occurrence Matrix (GLCM) on the GPU.

    # Arguments
    - `disc`: Discretized image stored on the GPU.
    - `gray_levels`: Gray levels.
    - `gpu_data`: GPU data container containing:
        - `gpu_data.img`: Original image stored on the GPU.
        - `gpu_data.mask`: ROI mask stored on the GPU.
        - `gpu_data.mask_indices`: Linear indices of valid ROI voxels.

    # Returns
    - `G`: Symmetric GLCM matrices on the CPU.

    # Notes
    - GLCM accumulation is performed on the GPU using atomic operations.
    - Symmetrization is performed on the CPU after GPU computation.
"""
function compute_glcm_gpu(disc::CuArray{Int},
    gray_levels::CuArray{Int},
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

    Ng = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
    lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    @cuda threads = CUDA_THREADS blocks = cld(Ng, CUDA_THREADS) lut_kernel!(gray_levels, lut, min_gl, Ng)

    mapped_disc = CUDA.zeros(Int, size(disc))
    Nx, Ny = size(mapped_disc)
    Nz = (dim == 3) ? size(mapped_disc, 3) : 1
    @cuda threads = CUDA_THREADS blocks = cld(length(disc), CUDA_THREADS) mapped_disc_kernel!(disc, mapped_disc, gpu_data.mask, length(disc), lut, min_gl)

    G_d = CUDA.zeros(Float64, Ng, Ng, length(dirs_x))

    n = length(gpu_data.mask_indices)
    num_dirs = length(dirs_x)

    blocks = (cld(n, CUDA_BLOCK_WIDTH_2D), cld(num_dirs, CUDA_BLOCK_HEIGHT_2D))
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = blocks glcm_kernel!(G_d, gpu_data.mask, gpu_data.mask_indices, mapped_disc, dirs_x, dirs_y, dirs_z, length(dirs_x), Nx, Ny, Nz, n)
    G_all = Array(G_d)

    for d in axes(G_all, 3)
        sym_sum = @view G_all[:, :, d]
        sym_sum .+= sym_sum'
    end
    return permutedims(G_all, (3, 1, 2))
end

function glcm_kernel_shmem!(G::CuDeviceArray{Float64},
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
    Ng::Int,
    num_valid::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= num_valid
        x, y, z = decode_xyz(mask_indices[i], Nx, Ny, Nz)
        if mask[x, y, z]
            i_disc = mapped_disc[x, y, z]
            @inbounds for j in 1:dirs_length
                dx = dirs_x[j];
                dy = dirs_y[j]
                dz = Nz > 1 ? dirs_z[j] : 0
                nx = x + dx;
                ny = y + dy;
                nz = z + dz
                if 1 <= nx <= Nx && 1 <= ny <= Ny && (Nz == 1 || (1 <= nz <= Nz))
                    if mask[nx, ny, nz]
                        j_disc = mapped_disc[nx, ny, nz]
                        CUDA.@atomic G[i_disc, j_disc, j] += 1.0
                    end
                end
            end
        end
    end
    return nothing

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

    lin_idx = mask_indices[i]

    # here we map a 1D index into 3D or 2D coordinates
    z = 1
    if Nz > 1
        z = fld(lin_idx - 1, Nx * Ny) + 1 # depth index
    end
    r = (lin_idx - 1) % (Nx * Ny)   # index inside 2d plane of size Nx * Ny
    y = fld(r, Nx) + 1              # row index from 1 to Ny
    x = (r % Nx) + 1                # column index from 1 to Nx

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
    CUDA.@atomic G[i_disc, j_disc, j] += 1.0

    return nothing
end