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
                              verbose=false)
    # Extracts radiomic features from the given image and mask. 
    # Parameters:
    - `img_input`: The input image (Array).
    - `mask_input`: The mask defining the region of interest (Array).
    - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
    - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
    - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `verbose`: If true, prints progress messages. 
    # Returns:
    - A dictionary where keys are the feature names and values are the calculated feature values.
    """
function extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                                            force_2d::Bool=false,
                                            force_2d_dimension::Int=1,
                                            n_bins::Union{Int,Nothing}=nothing,
                                            bin_width::Union{Float32,Nothing}=nothing,
                                            verbose::Bool=false)::Dict{String, Float32}

    total_start_time = time()
    total_time_accumulated = 0.0
    total_bytes_accumulated = 0

    if verbose
        println("Extracting radiomic features...")
        if !isnothing(n_bins)
            println("Using n_bins = $n_bins")
        elseif !isnothing(bin_width)
            println("Using bin_width = $bin_width")
        else
            println("Using default n_bins = 32")
        end
    end

    radiomic_features = Dict{String, Float32}()

    # Cast + prepare inputs
    img, mask, voxel_spacing = prepare_inputs(img_input, mask_input, voxel_spacing_input,
                                              force_2d, force_2d_dimension)

    # Sanity check
    result = @timed input_sanity_check(img, mask, verbose)
    total_time_accumulated += result.time
    total_bytes_accumulated += result.bytes
    if verbose
        println("Sanity check: $(result.time) sec, $(result.bytes / 1024^2) MiB")
    end

    if ndims(img) == 3
        # First order features
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

    # Work with 3d or 2d images - GLCM features
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

    if ndims(mask) == 2
        # 2D shape features
        result = @timed get_shape2d_features(mask, voxel_spacing, verbose)
        shape_2d_features = result.value
        merge!(radiomic_features, shape_2d_features)
        total_time_accumulated += result.time
        total_bytes_accumulated += result.bytes
        if verbose
            println("2D shape: $(result.time) sec, $(result.bytes / 1024^2) MiB")
            print_features("2D Shape Features", shape_2d_features)
        end

    elseif ndims(mask) == 3
        # 3D shape features
        result = @timed get_shape3d_features(mask, voxel_spacing; verbose=verbose)
        shape_3d_features = result.value
        merge!(radiomic_features, shape_3d_features)
        total_time_accumulated += result.time
        total_bytes_accumulated += result.bytes
        if verbose
            println("3D shape: $(result.time) sec, $(result.bytes / 1024^2) MiB")
            print_features("3D Shape Features", shape_3d_features)
        end

        # GLSZM features
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

        # NGTDM features
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

        # GLRLM features
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

        # GLDM features
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
Examples of usage:
# 1) With specific n_bins: 
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; n_bins=64, verbose=true);

# 2) With specific bin_width: 
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; bin_width=25.0f0, verbose=true);

# 3) With default setting: 
radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=true);
# Uses default bin_width (25)
"""

end