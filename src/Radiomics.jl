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

"""
    extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                              force_2d=false,
                              force_2d_dimension=1,
                              n_bins=nothing,
                              bin_width=nothing,
                              verbose=false,
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
                                   verbose::Bool=false,
                                   features::Vector{Symbol}=[:all])::Dict{String, Float32}

    total_start_time = time()
    total_time_accumulated = 0.0
    total_bytes_accumulated = 0

    # If features contains :all, compute all features
    compute_all = :all in features
    
    if verbose
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
            println("Using default n_bins = 32")
        end
    end

    radiomic_features = Dict{String, Float32}()

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

    # First order features (only for 3D images)
    if ndims(img) == 3 && (compute_all || :first_order in features)
        result = @timed get_first_order_features(img, mask, voxel_spacing, verbose)
        first_order_features = result.value
        merge!(radiomic_features, first_order_features)
        total_time_accumulated += result.time
        total_bytes_accumulated += result.bytes
        if verbose
            println("First order: $(result.time) sec, $(result.bytes / 1024^2) MiB")
            print_features("First Order Features", first_order_features)
        end
    end

    # GLCM features (2D or 3D)
    if compute_all || :glcm in features
        result = @timed get_glcm_features(img, mask, voxel_spacing; 
                                          n_bins=n_bins, 
                                          bin_width=bin_width, 
                                          verbose=verbose)
        glcm_features = result.value
        merge!(radiomic_features, glcm_features)
        total_time_accumulated += result.time
        total_bytes_accumulated += result.bytes
        if verbose
            println("GLCM: $(result.time) sec, $(result.bytes / 1024^2) MiB")
            print_features("GLCM Features", glcm_features)
        end
    end

    if ndims(mask) == 2
        # 2D shape features
        if compute_all || :shape2d in features
            result = @timed get_shape2d_features(mask, voxel_spacing, verbose)
            shape_2d_features = result.value
            merge!(radiomic_features, shape_2d_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("2D shape: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("2D Shape Features", shape_2d_features)
            end
        end

    elseif ndims(mask) == 3
        # 3D shape features
        if compute_all || :shape3d in features
            result = @timed get_shape3d_features(mask, voxel_spacing; verbose=verbose)
            shape_3d_features = result.value
            merge!(radiomic_features, shape_3d_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("3D shape: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("3D Shape Features", shape_3d_features)
            end
        end

        # GLSZM features
        if compute_all || :glszm in features
            result = @timed get_glszm_features(img, mask, voxel_spacing; 
                                          n_bins=n_bins, 
                                          bin_width=bin_width, 
                                          verbose=verbose)
            glszm_features = result.value
            merge!(radiomic_features, glszm_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("GLSZM: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("GLSZM Features", glszm_features)
            end
        end

        # NGTDM features
        if compute_all || :ngtdm in features
            result = @timed get_ngtdm_features(img, mask, voxel_spacing; 
                                          n_bins=n_bins, 
                                          bin_width=bin_width, 
                                          verbose=verbose)
            ngtdm_features = result.value
            merge!(radiomic_features, ngtdm_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("NGTDM: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("NGTDM Features", ngtdm_features)
            end
        end

        # GLRLM features
        if compute_all || :glrlm in features
            result = @timed get_glrlm_features(img, mask, voxel_spacing; 
                                          n_bins=n_bins, 
                                          bin_width=bin_width, 
                                          verbose=verbose)
            glrlm_features = result.value
            merge!(radiomic_features, glrlm_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("GLRLM: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("GLRLM Features", glrlm_features)
            end
        end

        # GLDM features
        if compute_all || :gldm in features
            result = @timed get_gldm_features(img, mask, voxel_spacing; 
                                          n_bins=n_bins, 
                                          bin_width=bin_width, 
                                          verbose=verbose)
            gldm_features = result.value
            merge!(radiomic_features, gldm_features)
            total_time_accumulated += result.time
            total_bytes_accumulated += result.bytes
            if verbose
                println("GLDM: $(result.time) sec, $(result.bytes / 1024^2) MiB")
                print_features("GLDM Features", gldm_features)
            end
        end
    end

    total_time_real = time() - total_start_time

    if verbose
        println("\n======================")
        println("Total features extracted: $(length(radiomic_features))")
        println("Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
        println("Real time (end-to-end): $(total_time_real) sec")
        println("Overhead: $(total_time_real - total_time_accumulated) sec")
        println("Total memory allocated: $(total_bytes_accumulated / 1024^2) MiB")
        println("======================\n")
        println("Radiomic features extraction completed.")
    end

    return radiomic_features
end

"""
# Examples:
    # Compute only GLCM features
    features = extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm], verbose=true);
    
    # Compute only first_order and shape3d
    features = extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:first_order, :shape3d], verbose=true);
    
    # Compute GLCM and GLSZM with specific bin_width
    features = extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm], 
                                        bin_width=25.0f0, 
                                        verbose=true);
    
    # Compute all features (default behavior)
    features = extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:all], verbose=true);
    # or simply:
    features = extract_radiomic_features(ct.raw, mask.raw, spacing, verbose=true);
    
    # Compute all texture features
    features = extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm, :glrlm, :gldm, :ngtdm], verbose=true);
    ```
"""

end
