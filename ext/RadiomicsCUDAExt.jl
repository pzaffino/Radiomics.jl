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
include("RadiomicsCUDAExt/glszm_features_gpu.jl")

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
function extract_radiomics_features_gpu(
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
    verbose::Bool=false)

    t_glcm_features = t_glszm_features = t_gldm_features = t_glrlm_features = t_ngtdm_features = nothing

    img_gpu = mask_gpu = mask_indices_gpu = nothing
    gpu_data = nothing
    img_gpu, mask_gpu, mask_indices_gpu = init_gpu(img, mask, verbose)
    gpu_data = GPUData(img_gpu, mask_gpu, mask_indices_gpu, nothing)

    if any(x -> x in (:glcm, :gldm, :glrlm, :ngtdm, :glszm), features) || compute_all
        gpu_data.texture_data = discretize_image_gpu(img, mask, gpu_data; n_bins=n_bins, bin_width=bin_width)
    end

    if compute_all || :glcm in features
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
        t_glcm_features = (result.value, result.time)
    end

    if compute_all || :glszm in features
        result = @timed CUDA.@sync get_glszm_features(
            img, mask, voxel_spacing;
            n_bins=n_bins,
            bin_width=bin_width,
            get_raw_matrices=get_raw_matrices,
            gpu_data=gpu_data,
            verbose=verbose
        )
        t_glszm_features = (result.value, result.time)
    end

    if compute_all || :ngtdm in features
        result = @timed CUDA.@sync get_ngtdm_features(
            img, mask, voxel_spacing;
            n_bins=n_bins,
            bin_width=bin_width,
            get_raw_matrices=get_raw_matrices,
            gpu_data=gpu_data,
            verbose=verbose
        )
        t_ngtdm_features = (result.value, result.time)
    end

    if compute_all || :glrlm in features
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
        t_glrlm_features = (result.value, result.time)
    end

    if compute_all || :gldm in features
        result = @timed CUDA.@sync get_gldm_features(
            img, mask, voxel_spacing;
            n_bins=n_bins,
            get_raw_matrices=get_raw_matrices,
            verbose=verbose,
            gpu_data=gpu_data
        )
        t_gldm_features = (result.value, result.time)
    end

    return (
        t_glcm_features,
        t_glszm_features,
        t_gldm_features,
        t_glrlm_features,
        t_ngtdm_features
    )
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
function calculate_diam2d_gpu(
    triangles::Vector{Radiomics.Triangle3D},
    verbose::Bool=false)
    verbose && println("[CUDA] Calculating 2D diameters on the GPU...")

    return calculate_diam2d(CuArray(triangles), verbose)
end

"""
    get_coefficients_gpu(mask_array::BitArray{2},
                                           spacing::Vector{Float64})

    # Arguments
    - `mask_array`: Binary mask.
    - `spacing`: Voxel spacing.

    # Returns
    Coefficients
"""
function get_coefficients_gpu(mask_array::BitArray{2}, spacing::Vector{Float64})
    return get_coefficients_gpu(CuArray(mask_array), CuArray(spacing))
end

@setup_workload begin
    if CUDA.functional()
        # Small synthetic data for precompilation warmup
        img_small = Float64.(reshape(1:1000, 10, 10, 10))
        mask_small = zeros(Float64, 10, 10, 10)
        mask_small[3:7, 3:7, 3:7] .= 1.0

        # Small 2D synthetic data for precompilation warmup
        img_small_2d = Float64.(reshape(1:100, 10, 10))
        mask_small_2d = zeros(Float64, 10, 10)
        mask_small_2d[3:7, 3:7] .= 1.0

        # 2D mask multi-label
        img_small_2d_multi = Float64.(reshape(1:100, 10, 10))
        mask_small_2d_multi = zeros(Float64, 10, 10)
        mask_small_2d_multi[3:7, 3:7] .= 1.0
        mask_small_2d_multi[6:8, 6:8] .= 2.0

        # Multi-label mask
        mask_multi = zeros(Float64, 10, 10, 10)
        mask_multi[2:4, 2:4, 2:4] .= 1.0
        mask_multi[6:8, 6:8, 6:8] .= 2.0

        spacing = [1.0, 1.0, 1.0]

        @compile_workload begin
            # 2D
            Radiomics.extract_radiomic_features(
                img_small_2d, mask_small_2d, spacing;
                keep_largest_only=true,
                use_gpu=true,
                verbose=false
            )

            # 2D mask multi-label
            Radiomics.extract_radiomic_features(
                img_small_2d_multi, mask_small_2d_multi, spacing;
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            #2D label
            Radiomics.extract_radiomic_features(
                img_small_2d_multi, mask_small_2d_multi, spacing;
                keep_largest_only=false,
                labels=[1, 2],
                use_gpu=true,
                verbose=false
            )

            # --- Default bin_width ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- n_bins ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                n_bins=32,
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- bin_width explicit ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                bin_width=25.0,
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- Weighting norms ---
            for wn in ["euclidean", "infinity", "manhattan", "no_weighting"]
                Radiomics.extract_radiomic_features(
                    img_small, mask_small, spacing;
                    weighting_norm=wn,
                    keep_largest_only=false,
                    use_gpu=true,
                    verbose=false
                )
            end

            # --- Selective features ---
            for feat in [
                [:glcm],
                [:first_order],
                [:shape3d],
                [:glszm],
                [:ngtdm],
                [:glrlm],
                [:gldm],
                [:glcm, :first_order],
                [:glcm, :glszm, :glrlm, :gldm, :ngtdm],
            ]
                Radiomics.extract_radiomic_features(
                    img_small, mask_small, spacing;
                    features=feat,
                    keep_largest_only=false,
                    use_gpu=true,
                    verbose=false
                )
            end

            # --- keep_largest_only = true ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                keep_largest_only=true,
                use_gpu=true,
                verbose=false
            )

            # --- features_std = true ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                features=[:glrlm],
                features_std=true,
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- Multi-label ---
            Radiomics.extract_radiomic_features(
                img_small, mask_multi, spacing;
                labels=[1, 2],
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- Single explicit label ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                labels=1,
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- get_raw_matrices ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                get_raw_matrices=true,
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )

            # --- 2D slice extraction ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                slices_2d=[(1, 5)],
                keep_largest_only=true,
                use_gpu=true,
                verbose=false
            )

            # --- Multiple slices ---
            Radiomics.extract_radiomic_features(
                img_small, mask_small, spacing;
                slices_2d=[(1, 5), (2, 5), (3, 5)],
                keep_largest_only=false,
                use_gpu=true,
                verbose=false
            )
        end
    end
end

end