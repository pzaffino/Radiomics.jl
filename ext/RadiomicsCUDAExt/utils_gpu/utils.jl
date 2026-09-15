CUDA_THREADS = 256

CUDA_BLOCK_WIDTH_2D = 16
CUDA_BLOCK_HEIGHT_2D = 16

CUDA_BLOCK_HEIGHT_3D = 8
CUDA_BLOCK_WIDTH_3D = 8
CUDA_BLOCK_DEPTH_3D = 4

mutable struct TextureData
    discretized_image::CuArray{Int}
    gray_levels::CuArray{Int}
    gl_lut::CuArray{Int}
    gray_levels_cpu::Array{Int}
    num_gl::Int
    max_gl::Int
    min_gl::Int
    n_bins::Union{Int,Nothing}
    bin_width::Union{Real,Nothing}
end

mutable struct GPUData
    img::CuArray{Float64}
    mask::CuArray{Bool}
    mask_indices::CuArray{Int}
    texture_data::Union{TextureData,Nothing}
end
struct GPUDict{T}
    key::CuArray{T,1}
    value::CuArray{T,1}
end

"""
    init_gpu(img_host::AbstractArray{Float64},
             mask_host::AbstractArray,
             verbose:Bool)::Tuple{CuArray{Float64},CuArray{Bool},CuArray{Int},Bool}

    Verifies that the CUDA.jl library and CUDA driver are installed and configured properly,
    and that the hardware is CUDA compatible.

    # Arguments
    - `img_host::AbstractArray{Float64}`: The input image (2D or 3D array) stored on the CPU
    - `mask_host::AbstractArray`: The binary mask defining the region of interest (same shape as `img_host`) stored on the CPU
    - `verbose::Bool`: If `true`, prints a compatibility confirmation message

    # Returns 
    If compatible:
    - `img_device::CuArray`: The input image stored on the GPU as `CuArray`
    - `mask_device::CuArray`: The binary mask stored on the GPU as `CuArray` 
    - `mask_indices::CuArray`: The vector of valid ROI indices 
    - `true`: flag indicating that GPU execution can proceed normally
    If not compatible:
    - `img_device`, `mask_device` and `mask_indices` are returned as `nothing`
    - `false`: Flag indicating that GPU execution cannot proceed
"""
function init_gpu(img_host::AbstractArray{Float64},
    mask_host::AbstractArray,
    verbose::Bool)::Tuple{CuArray{Float64},CuArray{Bool},CuArray{Int},Bool}
    compatible, errors = can_use_cuda()
    if compatible
        @info "current hardware is CUDA compatible. Please note that the first execution may take longer while CUDA kernels are initialized. For faster subsequent runs, keep this Julia process running and don't close it"

        img_device = CuArray(img_host)
        mask_device = CuArray(mask_host)
        mask_indices = CuArray(findall(vec(mask_host)))

        return img_device, mask_device, mask_indices, true
    else
        error_msg = errors * "Falling back to the CPU"
        @warn error_msg
        return nothing, nothing, nothing, false
    end
end

"""
    can_use_cuda()::Tuple{Bool,String}

    Checks for the correct installation and configuration of CUDA.jl and the CUDA driver 
    
    # Arguments

    # Returns
    - `compatible::Bool`: Flag indicating whether the current system is CUDA compatible
    - `errors::String`: Error messages describing any detected issues
"""
function can_use_cuda()::Tuple{Bool,String}
    errors = ""
    compatible = true

    if !CUDACore.functional()
        compatible = false
        errors += "- CUDA.jl does not appear to be functional and has not been installed or configured properly\n"
    end

    if !CUDACore.has_cuda()
        compatible = false
        errors += "- The CUDA driver has not been installed or the system does not have a CUDA compatible GPU. Please refer to https://developer.nvidia.com/cuda-downloads"
    end

    return compatible, errors
end

"""
    fits_block_shared_memory(bytes::Int, source::String, verbose::Bool)::Bool

Checks whether the requested shared memory allocation fits within the maximum
shared memory available per block.

# Arguments
- `bytes::Int`: Number of bytes of shared memory requested per block.
- `source::String`: Description of the kernel that's requesting shared memory
- `verbose::Bool`: If true, prints progress messages.

# Returns
- `use_shmem::Bool`: `true` if the requested allocation fits within the
  maximum shared memory per block, `false` otherwise
"""
function fits_block_shared_memory(bytes::Int, source::String, verbose::Bool)::Bool
    max_block_memory = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    use_shmem = bytes <= max_block_memory
    if verbose
        if use_shmem
            println("Requested shared memory allocation for $source is within the $max_block_memory bytes limit ($bytes bytes per block), using shared memory to minimize shared resource contention")
        else
            @warn "Shared memory allocation for $source exceeds the limit of $max_block_memory bytes because the ROI is too large. $bytes bytes per block requested. Falling back to global memory, execution may be slightly slower due to shared resource contention"
        end
    end
    return use_shmem
end

"""
    findall_gpu(mask_device::CuArray{Bool})::CuArray{Int}

    Scans the mask and extracts all valid ROI indices 

    # Arguments
    - `mask_device`: The binary mask defining the region of interest stored on the GPU

    # Returns 
    - `valid_idx`: The vector containing all valid ROI indices

    # Implementation
    The function reshapes `mask_device` into a 1D vector called `vec_mask`. Since GPU memory allocation can't be dynamic, the function needs to know allocation size beforehand.
    In this specific case, the function needs to allocate an array containing all valid ROI indices. To know how big this array will be, since we're dealing with a binary mask we sum all the elements in `mask_device`,
    the resulting sum will represent the number of useful voxels in `num_of_useful_voxels`. We use this value to allocate `valid_idx` which is the `CuArray` containing all valid ROI indices and has length `num_of_useful_voxels`.

    Since dynamic allocation is not allowed on the GPU, it's not possible to push an element into an array, to get around this we calculate the prefix sum of the mask so that every thread knows where it needs to write inside `valid_idx`
    and then call the kernel `findall_kernel!` to finally extract all indices.
    
    - Prefix sum and indexing example:
    The prefix sum of an array is an array where each element is the sum of all previous elements including itself
    vec_mask = [1, 0, 1, 1, 0]
    prefix_sum = [1, 1, 2, 3, 3]
    each element in `prefix_sum` is the position inside `valid_idx` where each thread will write if the mask in that specific position is true
"""
function findall_gpu(mask_device::CuArray{Bool})::CuArray{Int}
    vec_mask = vec(mask_device)
    num_of_useful_voxels = CUDA.sum(mask_device)
    valid_idx = CUDA.zeros(Int32, num_of_useful_voxels)
    prefix_sum = cumsum(Int32.(vec_mask))
    mask_length = length(vec_mask)

    @cuda threads = CUDA_THREADS blocks = cld(mask_length, CUDA_THREADS) findall_kernel!(vec_mask, prefix_sum, valid_idx, mask_length)

    CUDA.synchronize()

    return valid_idx
end

"""
    apply_mask(img::CuArray{Int},
               mask_indices::CuArray{Int})::CuArray{Int}
    Performs boolean indexing on the GPU and returns a 1D vector containing all the elements inside the ROI. CPU counterpart: roi = img[mask]

    # Arguments
    - `img`: The input image stored on the GPU as a `CuArray`
    - `mask_indices`: A vector of valid ROI indices 

    # Returns 
    - `roi`: 1D vector containing all the elements inside the ROI
"""
function apply_mask(img::CuArray{Int},
    mask_indices::CuArray{Int})::CuArray{Int}

    n = length(mask_indices)
    roi = CuArray{eltype(img)}(undef, n)

    @cuda threads = CUDA_THREADS blocks = cld(n, CUDA_THREADS) assign!(img, mask_indices, roi, n)

    return roi
end

"""
    discretize_image_gpu(gpu_data::GPUData;
                         n_bins::Union{Int,Nothing}=nothing,
                         bin_width::Union{<:Real,Nothing}=nothing)::Tuple{CuArray{Int},Int,CuArray{Int},Float64}
    
    Discretizes the input image for radiomics feature calculation. 
    Takes into account only the voxels within the provided mask.

    You can specify EITHER n_bins (number of bins) OR bin_width (width of each bin), but not both.
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins is calculated automatically
    - If neither is specified, defaults to bin_width=25.0

    This function is compatible with all radiomics features: GLCM, GLDM, GLRLM, GLSZM, NGTDM, etc.

    # Arguments:
        - `gpu_data`: A container storing GPU arrays:
            - `gpu_data.img`: The input image as a `CuArray`.
            - `gpu_data.mask`: The ROI mask as a `CuArray`.
            - `gpu_data.mask_indices`: Valid mask indices as a `CuArray`.
        - `n_bins`: The number of discrete gray levels (optional).
        - `bin_width`: The width of each bin (optional).

    # Returns:
        - `disc`: The discretized image as a `CuArray` of integers.
        - `n_bins_actual`: The actual number of discrete gray levels present.
        - `bin_width_used`: The bin width used for discretization.

"""
function discretize_image_gpu(img_cpu::AbstractArray{Float64},
    mask_cpu::BitArray,
    gpu_data::GPUData;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{<:Real,Nothing}=nothing,
    vmin::Union{Float64,Nothing}=nothing,
    vmax::Union{Float64,Nothing}=nothing,
    verbose_timing::Bool=true)::TextureData


    if length(gpu_data.mask_indices) == 0
        return zeros(Int, size(img)), 0, Int[], 0.0f0
    end

    if isnothing(vmin) || isnothing(vmax)
        vals = view(img_cpu, mask_cpu)
        vmin = minimum(vals)
        vmax = maximum(vals)
    end

    disc = CuArray{Int}(undef, size(gpu_data.img))

    if !isnothing(n_bins) && !isnothing(bin_width)
        error("Specify either n_bins or bin_width, not both.")
    elseif isnothing(n_bins) && isnothing(bin_width)
        bin_width = 25.0
    end

    n_of_indices = length(gpu_data.mask_indices)

    max_gl = CuArray([typemin(Int64)])
    min_gl = CuArray([typemax(Int64)])

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float64(n_bins)
        if bin_width_used ≈ 0.0
            bin_width_used = 1.0
        end

        inv_bin_width = 1.0 / bin_width_used

        blocks = cld(n_of_indices, CUDA_THREADS)
        @cuda threads = CUDA_THREADS blocks = blocks bin_nbins_kernel!(
            gpu_data.img, gpu_data.mask_indices, max_gl, min_gl,
            inv_bin_width, n_bins, vmin, disc, n_of_indices)
        n_bins_actual = n_bins
    else
        bin_width_used = bin_width
        inv_bin_width = 1.0 / bin_width_used
        bin_offset = Int(floor(vmin * inv_bin_width))

        blocks = cld(n_of_indices, CUDA_THREADS)
        @cuda threads = CUDA_THREADS blocks = blocks bin_width_kernel!(
            gpu_data.img, gpu_data.mask_indices, max_gl, min_gl,
            inv_bin_width, bin_offset, disc, n_of_indices)
        n_bins_actual = Int(floor((vmax - vmin) * inv_bin_width)) + 1
    end

    masked = apply_mask(disc, gpu_data.mask_indices)

    max_gl = Array(max_gl)[1]
    min_gl = Array(min_gl)[1]

    gray_levels = unique_gpu(masked, max_gl)
    gray_levels_cpu = Array(gray_levels)

    texture_data = TextureData(disc, gray_levels, create_lut(gray_levels_cpu, max_gl, min_gl), gray_levels_cpu, length(gray_levels), max_gl, min_gl, n_bins_actual, bin_width_used)

    return texture_data
end

function unique_gpu(img::CuArray{T}, max_gl::Int)::CuArray{T} where {T}
    values = CUDA.zeros(UInt32, max_gl)

    img_length = length(img)
    num_values = CUDA.zeros(Int, 1)

    # Mark which gray levels exist
    @cuda threads=CUDA_THREADS blocks=cld(img_length, CUDA_THREADS) mark_existing_values!(img, values, num_values, img_length,)

    num = Array(num_values)[1]

    uniques = CuArray{T}(undef, num)

    # Reuse counter for compacting the unique values
    CUDA.fill!(num_values, 0)

    @cuda threads=CUDA_THREADS blocks=cld(max_gl, CUDA_THREADS) assign_uniques_full!(values, uniques, num_values, max_gl,)

    return uniques
end


"""
    unique_gpu_shape(img::CuArray{T})::CuArray{T}

Compute the unique values of a `CuArray`

# Arguments
- `img`: A `CuArray` containing the values for which unique elements should be extracted.

# Returns
- `uniques`: A `CuArray` containing the unique values present in `img`.
"""
function unique_gpu_shape(img::CuArray{T}; verbose_timing::Bool=true)::CuArray{T} where T

    img = sort(img)
    n = length(img)

    is_boundary = CUDA.zeros(Int32, n)
    num_of_uniques = CuArray([0])
    @cuda threads = CUDA_THREADS blocks = cld(n, CUDA_THREADS) set_boundaries!(img, is_boundary, num_of_uniques)
    n_uniques_host = Array(num_of_uniques)[1]
    uniques = CuArray{T}(undef, n_uniques_host)
    counter = CuArray([0])

    @cuda threads = CUDA_THREADS blocks = cld(n, CUDA_THREADS) assign_uniques!(img, is_boundary, counter, uniques)
    return uniques
end

function create_lut(gray_levels::Array{Int}, max_gl::Int, min_gl::Int)
    lut = zeros(Int, max_gl - min_gl + 1)

    @inbounds for (i, gl) in enumerate(gray_levels)
        lut[Int(gl)-min_gl+1] = i
    end

    return lut
end