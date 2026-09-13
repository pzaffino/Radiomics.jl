module RadiomicsCUDAExt

using Radiomics
using CUDA
using PrecompileTools

include("RadiomicsCUDAExt/utils_gpu/utils.jl")
include("RadiomicsCUDAExt/utils_gpu/utils_kernels.jl")
include("RadiomicsCUDAExt/glcm_features_gpu.jl")
include("RadiomicsCUDAExt/shape_2D_features_gpu.jl")
include("RadiomicsCUDAExt/shape_3D_features_gpu.jl")
include("RadiomicsCUDAExt/ngtdm_features_gpu.jl")
include("RadiomicsCUDAExt/glrlm_features_gpu.jl")
include("RadiomicsCUDAExt/gldm_features_gpu.jl")

"""
    _compute_radiomics_impl(img, mask, voxel_spacing; 
                           n_bins, bin_width,
                           weighting_norm, verbose, 
                           keep_largest_only, compute_all, features, log_buffer)
    
    Internal function that handles parallel computation of radiomic features.
    Spawns separate threads for each feature category and collects results.
    
    # Parameters:
    - `img`: Preprocessed image array
    - `mask`: Preprocessed mask array  
    - `voxel_spacing`: Voxel spacing array
    - `voxel_count`: Number of voxels in the mask
    - `n_bins`: Number of bins for discretization
    - `bin_width`: Bin width
    - `weighting_norm`: Weighting norm for features
    - `verbose`: Print progress messages
    - `keep_largest_only`: Keep only largest connected component
    - `compute_all`: Compute all features or only selected ones
    - `features`: Vector of feature symbols to compute
    - `log_buffer`: Optional buffer for collecting log messages (for multi-label parallel processing)
    
    # Returns:
    - Tuple of (radiomic_features::Dict, total_time_accumulated::Float64)
"""
function Radiomics.extract_radiomics_features_gpu(
    features::Vector{Symbol},
    img::Array{Float64},
    mask::BitArray,
    voxel_spacing::Vector{Float64};
    n_bins::Union{Nothing,Int}=nothing,
    bin_width::Union{Nothing,Float64}=nothing,
    weighting_norm::Union{Nothing,String}=nothing,
    keep_largest_only::Bool=true,
    compute_all::Bool=true,
    features_std::Bool=false,
    get_raw_matrices::Bool=false,
    cuda_streams::Bool=false,
    verbose::Bool=false)

    t_glcm_features = t_gldm_features = t_glrlm_features = t_ngtdm_features = nothing

    img_gpu = mask_gpu = mask_indices_gpu = nothing
    gpu_data = nothing
    img_gpu, mask_gpu, mask_indices_gpu = init_gpu(img, mask, verbose)
    gpu_data = GPUData(img_gpu, mask_gpu, mask_indices_gpu, nothing)

    if any(x -> x in (:glcm, :gldm, :glrlm, :ngtdm), features) || compute_all
        gpu_data.texture_data = discretize_image_gpu(img, mask, gpu_data; n_bins=n_bins, bin_width=bin_width)
    end

    if cuda_streams
        if CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_CONCURRENT_KERNELS) == 1
            @info "The active GPU supports concurrent kernel execution. However, enabling cuda_streams does not guarantee that kernels will execute concurrently and may lead to GPU saturation, slowing down execution. It is recommended to use CUDA streams on a GPU with sufficient resources (streaming multiprocessors, available registers per SM, shared memeory per SM)."
        else
            @warn "The active GPU does not support concurrent kernel execution. Disabling CUDA streams"
            cuda_streams = false
        end
    end

    if compute_all || :glcm in features
        t_glcm_features = Threads.@spawn begin
            if cuda_streams
                glcm_stream = CUDA.CuStream()

                CUDA.stream!(glcm_stream) do
                    result = @timed get_glcm_features(
                        img, mask, voxel_spacing;
                        n_bins=n_bins,
                        bin_width=bin_width,
                        weighting_norm=weighting_norm,
                        features_std=features_std,
                        get_raw_matrices=get_raw_matrices,
                        gpu_data=gpu_data,
                        verbose=verbose
                    )
                    CUDA.synchronize(glcm_stream)
                    (result.value, result.time)
                end
            else
                result = @timed CUDA.@sync get_glcm_features(
                    img, mask, voxel_spacing;
                    n_bins=n_bins,
                    bin_width=bin_width,
                    weighting_norm=weighting_norm,
                    features_std=features_std,
                    get_raw_matrices=get_raw_matrices,
                    gpu_data=gpu_data,
                    verbose=verbose
                )
                (result.value, result.time)
            end
        end
    end

    if compute_all || :ngtdm in features
        t_ngtdm_features = Threads.@spawn begin
            if cuda_streams
                ngtdm_stream = CUDA.CuStream()
                CUDA.stream!(ngtdm_stream) do
                    result = @timed get_ngtdm_features(
                        img, mask, voxel_spacing;
                        n_bins=n_bins,
                        bin_width=bin_width,
                        get_raw_matrices=get_raw_matrices,
                        gpu_data=gpu_data,
                        verbose=verbose
                    )

                    CUDA.synchronize(ngtdm_stream)
                    (result.value, result.time)
                end
            else
                result = @timed CUDA.@sync get_ngtdm_features(
                    img, mask, voxel_spacing;
                    n_bins=n_bins,
                    bin_width=bin_width,
                    get_raw_matrices=get_raw_matrices,
                    gpu_data=gpu_data,
                    verbose=verbose
                )
                (result.value, result.time)
            end
        end
    end


    if compute_all || :glrlm in features
        t_glrlm_features = Threads.@spawn begin
            if cuda_streams
                glrlm_stream = CUDA.CuStream()
                CUDA.stream!(glrlm_stream) do
                    result = @timed get_glrlm_features(
                        img,
                        mask,
                        voxel_spacing;
                        n_bins=n_bins,
                        bin_width=bin_width,
                        features_std=features_std,
                        weighting_norm=weighting_norm,
                        get_raw_matrices=get_raw_matrices,
                        gpu_data=gpu_data,
                        verbose=verbose
                    )
                    CUDA.synchronize(glrlm_stream)
                    (result.value, result.time)
                end
            else
                result = @timed CUDA.@sync get_glrlm_features(
                    img,
                    mask,
                    voxel_spacing;
                    n_bins=n_bins,
                    bin_width=bin_width,
                    features_std=features_std,
                    weighting_norm=weighting_norm,
                    get_raw_matrices=get_raw_matrices,
                    gpu_data=gpu_data,
                    verbose=verbose
                )
                (result.value, result.time)
            end
        end
    end


    if compute_all || :gldm in features
        t_gldm_features = Threads.@spawn begin
            if cuda_streams
                gldm_stream = CUDA.CuStream()

                CUDA.stream!(gldm_stream) do
                    result = @timed get_gldm_features(
                        img, mask, voxel_spacing;
                        n_bins=n_bins,
                        bin_width=bin_width,
                        get_raw_matrices=get_raw_matrices,
                        verbose=verbose,
                        gpu_data=gpu_data
                    )
                    CUDA.synchronize(gldm_stream)
                    (result.value, result.time)
                end
            else
                result = @timed CUDA.@sync get_gldm_features(
                    img, mask, voxel_spacing;
                    n_bins=n_bins,
                    bin_width=bin_width,
                    get_raw_matrices=get_raw_matrices,
                    verbose=verbose,
                    gpu_data=gpu_data
                )
                (result.value, result.time)
            end
        end
    end

    if compute_all || :glcm in features
        t_glcm_features = fetch(t_glcm_features)
    end

    if compute_all || :ngtdm in features
        t_ngtdm_features = fetch(t_ngtdm_features)
    end

    if compute_all || :glrlm in features
        t_glrlm_features = fetch(t_glrlm_features)
    end

    if compute_all || :gldm in features
        t_gldm_features = fetch(t_gldm_features)
    end

    return t_glcm_features, t_gldm_features, t_glrlm_features, t_ngtdm_features


end

"""
    Radiomics.calculate_diam2d_gpu(triangles::Vector{Radiomics.Triangle3D},
                                   verbose::Bool=false)

    # Arguments
    - `triangles`: Vector of 3D triangles.
    - `verbose`: Flag used to print progress information.

    # Returns
    2D diameters
"""
function Radiomics.calculate_diam2d_gpu(
    triangles::Vector{Radiomics.Triangle3D},
    verbose::Bool=false)
    verbose && println("[CUDA] Calculating 2D diameters on the GPU...")

    return calculate_diam2d_gpu(CuArray(triangles), verbose)
end

"""
    Radiomics.get_coefficients_gpu_wrapper(mask_array::BitArray{2},
                                           spacing::Vector{Float64})

    # Arguments
    - `mask_array`: Binary mask.
    - `spacing`: Voxel spacing.

    # Returns
    Coefficients
"""
function Radiomics.get_coefficients_gpu_wrapper(mask_array::BitArray{2}, spacing::Vector{Float64})
    return get_coefficients_gpu(CuArray(mask_array), CuArray(spacing))
end

"""
    Radiomics.max_dist2_gpu_wrapper(h::Int,
                                    hull::Vector{Tuple{Float64,Float64}})

    # Arguments
    - `h`: Number of points in the hull.
    - `hull`: Holl

    # Returns
    Maximum squared distance
"""
function Radiomics.max_dist2_gpu_wrapper(h::Int, hull::Vector{Tuple{Float64,Float64}})::Float64
    hull = CuArray(hull)
    max_dist2 = CuArray([0.0])
    blocks = (cld(h, 16), cld(h, 16))
    @cuda threads=(16, 16) blocks=blocks max_dist!(hull, max_dist2, h)
    return Array(max_dist2)[1]
end
@setup_workload begin
    img_small = Float64.(reshape(1:1000, 10, 10, 10))
    mask_small = zeros(Float64, 10, 10, 10)
    mask_small[3:7, 3:7, 3:7] .= 1.0

    img_small_gpu = reshape(Float64.(1:1000), 10, 10, 10)
    mask_small_gpu = zeros(Float64, 10, 10, 10)
    mask_small_gpu[1:6, 1:6, 1:6] .= 1.0
    mask_cpu = BitArray(mask_small_gpu .!= 0.0)

    img = CuArray(img_small_gpu)
    mask = CuArray(mask_cpu)
    mask_indices = CuArray(findall(vec(mask_cpu)))

    gpu_data = GPUData(img, mask, mask_indices, nothing)

    spacing = [1.0, 1.0, 1.0]

    @compile_workload begin
        texture_data = discretize_image_gpu(img_small, mask_cpu, gpu_data)
        gpu_data.texture_data = texture_data

        compute_glcm_gpu(gpu_data.texture_data.discretized_image, gpu_data)
        compute_gldm_gpu(gpu_data.texture_data.discretized_image, gpu_data.mask, gpu_data.mask_indices, gpu_data.texture_data.gray_levels, gpu_data.texture_data.gl_lut, gpu_data.texture_data.num_gl, gpu_data.texture_data.max_gl, gpu_data.texture_data.min_gl, 1)
        compute_glrlm_gpu(gpu_data.texture_data.discretized_image, gpu_data.mask, gpu_data.mask_indices, gpu_data.texture_data.gl_lut, gpu_data.texture_data.num_gl, gpu_data.texture_data.min_gl)
        compute_ngtdm_gpu(gpu_data.texture_data.discretized_image, gpu_data.mask, gpu_data.mask_indices, gpu_data.texture_data.gray_levels, gpu_data.texture_data.gray_levels_cpu, gpu_data.texture_data.gl_lut, gpu_data.texture_data.num_gl, gpu_data.texture_data.max_gl, gpu_data.texture_data.min_gl)

        Radiomics.get_shape3d_features(mask_cpu, spacing; verbose=false, keep_largest_only=false, use_gpu=true)
    end
end

end