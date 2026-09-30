"""
    find_root(parent, x::Int)
# Arguments
- `parent`: Parent array representing the forest
- `x`: element of interest

# Returns
- `Int`: Index of the root element of the set containing `x`
"""
@inline function find_root(parent, x::Int)
    @inbounds while true
        p = parent[x]
        p == x && return x
        x = p
    end
end

"""
    union_roots!(parent, a::Int, b::Int)

Merge the sets  `a` and `b`

# Arguments
- `parent`: Parent array representing the forest
- `a`: First element
- `b`: Second element

# Returns
- `nothing`
"""
@inline function union_roots!(parent, a::Int, b::Int)
    while true
        ra = find_root(parent, a)
        rb = find_root(parent, b)
        ra == rb && return
        ra > rb && ((ra, rb) = (rb, ra))
        old = CUDA.atomic_cas!(pointer(parent, rb), rb, ra)
        old == rb && return
    end
end

"""
    initialization!(parent::CuDeviceArray{Int}, img_length::Int)

# Arguments
- `parent`: Array storing the parent index for each voxel
- `img_length`: Number of voxels in the image

# Returns
- `nothing`.

"""
function initialization!(parent::CuDeviceArray{Int}, img_length::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= img_length
        @inbounds parent[i] = i
    end
    return nothing
end


"""
    union_kernel!(
        parent::CuDeviceArray{Int},
        img::CuDeviceArray{Int},
        Nx::Int,
        Ny::Int,
        Nz::Int,
        img_length::Int
    )

Create connected components of equal gray level

# Arguments
- `parent`: Array storing the parent index for each voxel
- `img`: Discretized image stored on the GPU
- `Nx::Int`: Image width
- `Ny::Int`: Image height
- `Nz::Int`: Image depth
- `img_length`: Total number of voxels in the image

# Returns
- `nothing`
"""
function union_kernel!(parent::CuDeviceArray{Int},
    img::CuDeviceArray{Int},
    Nx::Int, Ny::Int, Nz::Int, img_length::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i > img_length && return nothing

    @inbounds begin
        g = img[i]
        g == 0 && return nothing

        x, y, z = decode_xyz(i, Nx, Ny, Nz)

        dz_max = Nz > 1 ? 1 : 0
        for dz in 0:dz_max, dy in -1:1, dx in -1:1
            (dz == 0 && (dy < 0 || (dy == 0 && dx <= 0))) && continue
            xx = x + dx
            yy = y + dy
            zz = z + dz
            (1 <= xx <= Nx && 1 <= yy <= Ny && zz <= Nz) || continue
            j = xx + (yy - 1) * Nx + (zz - 1) * Nx * Ny
            if img[j] == g
                union_roots!(parent, i, j)
            end
        end
    end
    return nothing
end

"""
    flatten_kernel!(parent::CuDeviceVector{Int}, n::Int)

Replaces each element of the parent array with the root of its connected component

# Arguments
- `parent`: Array storing the parent index for each voxel
- `n`: Number of elements in `parent`

# Returns
- `nothing`
"""
function flatten_kernel!(parent::CuDeviceVector{Int}, n::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i > n && return nothing
    @inbounds parent[i] = find_root(parent, i)
    return nothing
end

"""
    count_zone_sizes!(
        sizes::CuDeviceVector{Int},
        parent::CuDeviceVector{Int},
        img::CuDeviceArray{Int},
        n::Int
    )

Count the number of voxels belonging to each connected component

# Arguments
- `sizes`: Array storing the size of each connected component
- `parent`: Parent array containing the component root for each voxel
- `img`: Discretized image stored on the GPU
- `n`: Number of voxels

# Returns
- `nothing`.
"""
function count_zone_sizes!(sizes::CuDeviceVector{Int},
    parent::CuDeviceVector{Int},
    img::CuDeviceArray{Int},
    n::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i > n && return nothing

    @inbounds if img[i] != 0
        CUDA.atomic_add!(pointer(sizes, parent[i]), 1)
    end
    return nothing
end

"""
    glszm_kernel!(
        P_glszm::CuDeviceMatrix{Int},
        img::CuDeviceArray{Int},
        parent::CuDeviceVector{Int},
        sizes::CuDeviceVector{Int},
        lut::CuDeviceVector{Int},
        min_gl::Int,
        img_length::Int,
        num_gl::Int
    )

Creates the GLSZM

# Arguments
- `P_glszm`: Matrix containing the GLSZM
- `img`: Discretized image stored on the GPU
- `parent`: Parent array
- `sizes`: Array containing the size of each connected component.
- `lut`: Gray level look up table
- `min_gl`: Minimum gray level in the discretized image
- `img_length`: Number of voxels
- `num_gl`: Number of gray levels in the discrized image

# Returns
- `nothing`
"""
function glszm_kernel!(P_glszm::CuDeviceMatrix{Int},
    img::CuDeviceArray{Int},
    parent::CuDeviceVector{Int},
    sizes::CuDeviceVector{Int},
    lut::CuDeviceVector{Int},
    min_gl::Int,
    img_length::Int,
    num_gl::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i > img_length && return nothing

    parent[i] != i && return nothing
    g = img[i]
    g == 0 && return nothing
    k = g - min_gl + 1
    (k < 1 || k > length(lut)) && return nothing
    row = lut[k]

    row == 0 && return nothing
    s = sizes[i]
    s <= 0 && return nothing
    CUDA.atomic_add!(pointer(P_glszm, row + (s - 1) * num_gl), 1)
    return nothing
end

"""
    compute_glszm_gpu(
        discretized_img::CuArray{Int},
        mask::CuArray{Bool},
        mask_indices::CuArray{Int},
        gray_levels::CuArray{Int},
        gray_levels_cpu::Array{Int},
        gl_lut::CuArray{Int},
        num_gl::Int,
        max_gl::Int,
        min_gl::Int
    )::Matrix{Int}

Calculates and returns a dictionary of GLSZM (Gray Level Size Zone Matrix) features.


# Arguments
- `discretized_img`: Discretized image stored on the GPU
- `mask`: Binary ROI mask stored on the GPU
- `gl_lut`: Gray level look up table
- `num_gl`: Number of gray levels
- `min_gl`: Minimum gray level in the discretized image

# Returns
- `Matrix{Int}`: GLSZM matrix transferred from the GPU to the CPU
"""
function compute_glszm_gpu(discretized_img::CuArray{Int},
    mask::CuArray{Bool},
    gl_lut::CuArray{Int},
    num_gl::Int,
    min_gl::Int)::Matrix{Int}

    dim = ndims(discretized_img)
    Nx, Ny = size(discretized_img)
    Nz = (dim == 3) ? size(discretized_img, 3) : 1
    img_length = length(discretized_img)

    parent = CuArray{Int}(undef, img_length)
    sizes = CUDA.zeros(Int, img_length)

    img = size(mask) == size(discretized_img) ?
          ifelse.(mask, discretized_img, 0) : discretized_img

    blocks = cld(img_length, CUDA_THREADS)
    @cuda threads=CUDA_THREADS blocks=blocks initialization!(parent, img_length)
    @cuda threads=CUDA_THREADS blocks=blocks union_kernel!(parent, img, Nx, Ny, Nz, img_length)
    @cuda threads=CUDA_THREADS blocks=blocks flatten_kernel!(parent, img_length)
    @cuda threads=CUDA_THREADS blocks=blocks count_zone_sizes!(sizes, parent, img, img_length)

    max_size = max(maximum(sizes), 1)
    P_glszm = CUDA.zeros(Int, num_gl, max_size)

    @cuda threads=CUDA_THREADS blocks=blocks glszm_kernel!(P_glszm, img, parent, sizes, gl_lut, min_gl, img_length, num_gl)

    return Array(P_glszm)
end

function get_glszm_features(img::AbstractArray{Float64},
    mask::BitArray,
    voxel_spacing::Vector{Float64};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    get_raw_matrices::Bool=false,
    verbose::Bool=false,
    gpu_data::GPUData)::Dict{String,Any}

    verbose && println("Calculating GLSZM matrix (GPU)...")

    P_glszm = compute_glszm_gpu(
        gpu_data.texture_data.discretized_image,
        gpu_data.mask,
        gpu_data.texture_data.gl_lut,
        gpu_data.texture_data.num_gl,
        gpu_data.texture_data.min_gl
    )

    P_glszm, _ = Radiomics.calculate_glszm_matrix(
        [0],
        mask,
        verbose,
        P_glszm,
        gpu_data.texture_data.gray_levels_cpu
    )

    return Radiomics.get_glszm_features(
        img,
        mask,
        voxel_spacing,
        n_bins=n_bins,
        bin_width=bin_width,
        get_raw_matrices=get_raw_matrices,
        verbose=verbose,
        P_glszm=P_glszm,
        gray_levels=gpu_data.texture_data.gray_levels_cpu
    )
end