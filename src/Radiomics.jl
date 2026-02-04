using Base.Threads

module Radiomics

include("utils.jl")
include("glcm_features.jl")
include("first_order_features.jl")
include("shape_2D_features.jl")
include("shape_3D_features.jl")
include("glszm_features.jl")
include("ngtdm_features.jl")
include("glrlm_features.jl")
include("gldm_features.jl")
include("diagnosis.jl")

using JSON3
using Pkg

"""
    extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                              force_2d=false,
                              force_2d_dimension=1,
                              n_bins=nothing,
                              bin_width=nothing,
                              weighting_norm=nothing,
                              verbose=false,
                              keep_largest_only=true,
                              features=[:all])
    
    Extracts radiomic features from the given image and mask.
    
    # Parameters:
    - `img_input`: The input image (Array).
    - `mask_input`: The mask defining the region of interest (Array).
    - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
    - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
    - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `verbose`: If true, prints progress messages.
    - `sample_rate`: The sample rate for feature extraction (optional).
    - `keep_largest_only`: If true, keeps only the largest connected component for 3D shape features (default: true).
    - `features`: Array of symbols specifying which features to compute. 
                 Options: :first_order, :glcm, :shape2d, :shape3d, :glszm, :ngtdm, :glrlm, :gldm, :all
                 Use [:all] to compute all features (default).
    
    # Returns:
    - A dictionary where keys are the feature names and values are the calculated feature values.
"""
function extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
    force_2d::Bool=false,
    force_2d_dimension::Int=1,
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    verbose::Bool=false,
    sample_rate::Float64=0.03,
    keep_largest_only::Bool=true,
    features::Vector{Symbol}=[:all])::Dict{String,Any}

    total_start_time = time()
    total_time_accumulated = 0.0 
    total_bytes_accumulated = 0

    # If features contains :all, compute all features
    compute_all = :all in features

    if verbose
        println("Active threads: ", Threads.nthreads())
        println("Extracting radiomic features...")
        if compute_all
            println("Computing ALL features")
        else
            println("Computing selected features: ", join(string.(features), ", "))
        end
        if !isnothing(n_bins)
            println("Using n_bins = $n_bins")
        elseif !isnothing(bin_width)
            println("Using bin_width = $bin_width")
        else
            println("Using default bin_width = 25.0")
        end
        if sample_rate != 0.03
            println("Using explicit sample_rate = $sample_rate")
        end
    end

    if isnothing(n_bins) && sum(mask_input) > 0
        control(img_input, mask_input, bin_width)
    end

    radiomic_features = Dict{String,Any}()

    # Cast and prepare inputs
    img, mask, voxel_spacing = prepare_inputs(img_input, mask_input, voxel_spacing_input,
        force_2d, force_2d_dimension)

    # Sanity check
    result = @timed input_sanity_check(img, mask, verbose)
    total_time_accumulated += result.time
    total_bytes_accumulated += result.bytes
    if verbose
        println("Sanity check: $(result.time) sec, $(result.bytes / 1024^2) MiB")
    end

    # Run threads

    if ndims(img) == 3 && (compute_all || :shape3d in features)
        t_sphape3d_features = Threads.@spawn @timed get_shape3d_features(mask, voxel_spacing; verbose=verbose, sample_rate=sample_rate, keep_largest_only=keep_largest_only)
    end

    # Run in a separate thread extraction of GLCM features (2D or 3D)
    if compute_all || :glcm in features
        t_glcm_features = Threads.@spawn @timed get_glcm_features(img, mask, voxel_spacing;
            n_bins=n_bins,
            bin_width=bin_width,
            weighting_norm=weighting_norm,
            verbose=verbose)
    end

    if ndims(mask) == 2
        # Run in a separate thread extraction of 2D shape features
        if compute_all || :shape2d in features
            t_shape2d_features = Threads.@spawn @timed get_shape2d_features(mask, voxel_spacing, verbose)
        end

    elseif ndims(mask) == 3
        # Run in a separate thread extraction of 3D shape features
        if compute_all || :first_order in features
            t_first_order_features = Threads.@spawn @timed get_first_order_features(img, mask, voxel_spacing; n_bins=n_bins, bin_width=bin_width, verbose=verbose)
        end

        # Run in a separate thread extraction of GLSZM features
        if compute_all || :glszm in features
            t_glszm_features = Threads.@spawn @timed get_glszm_features(img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                verbose=verbose)
        end

        # Run in a separate thread extraction of NGTDM features
        if compute_all || :ngtdm in features
            t_ngtdm_features = Threads.@spawn @timed get_ngtdm_features(img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                verbose=verbose)
        end

        # Run in a separate thread extraction of GLRLM features
        if compute_all || :glrlm in features
            t_glrlm_features = Threads.@spawn @timed get_glrlm_features(img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                weighting_norm=weighting_norm,
                verbose=verbose)
        end

        # Run in a separate thread extraction of GLDM features
        if compute_all || :gldm in features
            t_gldm_features = Threads.@spawn @timed get_gldm_features(img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                verbose=verbose)
        end
    end

    # Retrieve data from worker threads

    # First order features (only for 3D images)
    if ndims(img) == 3 && (compute_all || :first_order in features)
        results_first_order = fetch(t_first_order_features)
        first_order_features = results_first_order.value
        merge!(radiomic_features, first_order_features)
        total_time_accumulated += results_first_order.time
        total_bytes_accumulated += results_first_order.bytes
        if verbose
            println("First order: $(results_first_order.time) sec, $(results_first_order.bytes / 1024^2) MiB")
            print_features("First Order Features", first_order_features)
        end
    end

    # GLCM features (2D or 3D)
    if compute_all || :glcm in features
        results_glcm = fetch(t_glcm_features)
        glcm_features = results_glcm.value
        merge!(radiomic_features, glcm_features)
        total_time_accumulated += results_glcm.time
        total_bytes_accumulated += results_glcm.bytes
        if verbose
            println("GLCM: $(results_glcm.time) sec, $(results_glcm.bytes / 1024^2) MiB")
            print_features("GLCM Features", glcm_features)
        end
    end

    if ndims(mask) == 2
        # 2D shape features
        if compute_all || :shape2d in features
            results_shape2d = fetch(t_shape2d_features)
            shape_2d_features = results_shape2d.value
            merge!(radiomic_features, shape_2d_features)
            total_time_accumulated += results_shape2d.time
            total_bytes_accumulated += results_shape2d.bytes
            if verbose
                println("2D shape: $(results_shape2d.time) sec, $(results_shape2d.bytes / 1024^2) MiB")
                print_features("2D Shape Features", shape_2d_features)
            end
        end

    elseif ndims(mask) == 3
        # GLSZM features
        if compute_all || :glszm in features
            results_glszm = fetch(t_glszm_features)
            glszm_features = results_glszm.value
            merge!(radiomic_features, glszm_features)
            total_time_accumulated += results_glszm.time
            total_bytes_accumulated += results_glszm.bytes
            if verbose
                println("GLSZM: $(results_glszm.time) sec, $(results_glszm.bytes / 1024^2) MiB")
                print_features("GLSZM Features", glszm_features)
            end
        end

        # NGTDM features
        if compute_all || :ngtdm in features
            results_ngtdm = fetch(t_ngtdm_features)
            ngtdm_features = results_ngtdm.value
            merge!(radiomic_features, ngtdm_features)
            total_time_accumulated += results_ngtdm.time
            total_bytes_accumulated += results_ngtdm.bytes
            if verbose
                println("NGTDM: $(results_ngtdm.time) sec, $(results_ngtdm.bytes / 1024^2) MiB")
                print_features("NGTDM Features", ngtdm_features)
            end
        end

        # GLRLM features
        if compute_all || :glrlm in features
            results_glrlm = fetch(t_glrlm_features)
            glrlm_features = results_glrlm.value
            merge!(radiomic_features, glrlm_features)
            total_time_accumulated += results_glrlm.time
            total_bytes_accumulated += results_glrlm.bytes
            if verbose
                println("GLRLM: $(results_glrlm.time) sec, $(results_glrlm.bytes / 1024^2) MiB")
                print_features("GLRLM Features", glrlm_features)
            end
        end

        # GLDM features
        if compute_all || :gldm in features
            results_gldm = fetch(t_gldm_features)
            gldm_features = results_gldm.value
            merge!(radiomic_features, gldm_features)
            total_time_accumulated += results_gldm.time
            total_bytes_accumulated += results_gldm.bytes
            if verbose
                println("GLDM: $(results_gldm.time) sec, $(results_gldm.bytes / 1024^2) MiB")
                print_features("GLDM Features", gldm_features)
            end
        end

        # 3D shape features
        if compute_all || :shape3d in features
            results_shape3d = fetch(t_sphape3d_features)
            shape_3d_features = results_shape3d.value
            merge!(radiomic_features, shape_3d_features)
            total_time_accumulated += results_shape3d.time
            total_bytes_accumulated += results_shape3d.bytes
            if verbose
                println("3D shape: $(results_shape3d.time) sec, $(results_shape3d.bytes / 1024^2) MiB")
                print_features("3D Shape Features", shape_3d_features)
            end
        end
        
    end

    total_time_real = time() - total_start_time

    if verbose
        println("\n======================")
        println("Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
        println("Real time (end-to-end): $(total_time_real) sec")
        println("Overhead: $(total_time_real - total_time_accumulated) sec")
        println("Total memory allocated: $(total_bytes_accumulated / 1024^2) MiB")
        diagnosis_features = get_diagnosis_features(sample_rate, bin_width, voxel_spacing,total_time_real, 
                                                    total_bytes_accumulated, weighting_norm, n_bins, keep_largest_only,
                                                    img, mask)
        merge!(radiomic_features, diagnosis_features)
        print_features_diagnosis("Diagnosis Features", diagnosis_features)
        println("---------------------")
        println("Total features extracted: $(length(radiomic_features))")
        println("Radiomic features extraction completed.")
        println("======================")
    end

    return radiomic_features
end

"""
    Wrapper function to be exposed in the C shared library
"""
# Global Buffer globale to avoid the garbage collector (for shared library)
const LAST_JSON_RESULT = Ref{String}("")

Base.@ccallable function c_extract_radiomic_features(
    img_ptr::Ptr{Float32},
    img_size_x::Int64, img_size_y::Int64, img_size_z::Int64,
    mask_ptr::Ptr{Float32},
    spacing_x::Float64, spacing_y::Float64, spacing_z::Float64,
    n_bins::Int64
)::Cstring
    global LAST_JSON_RESULT
    try
        # Prepare inputs for the main function
        dims = (Int(img_size_x), Int(img_size_y), Int(img_size_z))
        spacing = [spacing_x, spacing_y, spacing_z]
        bins_val = n_bins <= 0 ? nothing : Int(n_bins)

        img = unsafe_wrap(Array, img_ptr, dims)
        mask = unsafe_wrap(Array, mask_ptr, dims)

        # Call the main function
        c_features_dict = extract_radiomic_features(
            img, mask, spacing;
            n_bins = bins_val,
            verbose = false
        )

        # Save the features in the global buffer
        LAST_JSON_RESULT[] = JSON3.write(c_features_dict) * "\0"
        return pointer(LAST_JSON_RESULT[])

    catch e
        @error "Error during feature extration step" exception=(e, catch_backtrace())

        err_msg = "{\"error\": \"$e\"}\0"
        LAST_JSON_RESULT[] = err_msg
        return pointer(LAST_JSON_RESULT[])
    end
end

"""
# Examples:
    # Compute only GLCM features
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm], verbose=true);
    
    # Compute only first_order and shape3d
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:first_order, :shape3d], verbose=true);
    
    # Compute GLCM and GLSZM with specific bin_width
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm], 
                                        bin_width=25.0f0, 
                                        verbose=true);
    
    # Compute all features (default behavior)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:all], verbose=true);
    # or simply:
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing, verbose=true);
    
    # Compute all texture features
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm, :glrlm, :gldm, :ngtdm], verbose=true);
    
    # Compute with sample_rate personalzed 
    features = Radiomics.extract_radiomic_features(img, mask, spacing; sample_rate=1.0, verbose=true)

    # Compute with keep_largest_only personalzed 
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false);

    # Compute with weighting_norm personalzed 
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, weighting_norm="euclidean");
"""

end
