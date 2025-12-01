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
    if verbose
        println("Extracting ONLY GLCM radiomic features...")
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
    sanity_check_start_time = time()
    input_sanity_check(img, mask, verbose)
    if verbose
        println("Sanity check time = $(time() - sanity_check_start_time) sec")
    end

    if ndims(img) == 3
        # First order features
        first_order_start_time = time()
        first_order_features = get_first_order_features(img, mask, voxel_spacing, verbose)
        merge!(radiomic_features, first_order_features)
        if verbose
            println("First order feature extraction time = $(time() - first_order_start_time) sec")
            print_features("First Order Features", first_order_features)
        end
    end

    # Work with 3d or 2d images
    glcm_start_time = time()
    glcm_features = get_glcm_features(img, mask, voxel_spacing; 
                                      n_bins=n_bins, 
                                      bin_width=bin_width, 
                                      verbose=verbose)
    merge!(radiomic_features, glcm_features)
    if verbose
        println("GLCM feature extraction time = $(time() - glcm_start_time) sec")
        print_features("GLCM Features", glcm_features)
        println("Total GLCM features extracted: $(length(radiomic_features))")
    end

    if ndims(mask) == 2
        # 2D shape features
        shape_2d_start_time = time()
        shape_2d_features = get_shape2d_features(mask, voxel_spacing, verbose)
        merge!(radiomic_features, shape_2d_features)
        if verbose
            println("2D shape feature extraction time = $(time() - shape_2d_start_time) sec")
            print_features("2D Shape Features", shape_2d_features)
        end

    elseif ndims(mask) == 3
        # 3D shape features
        shape_3d_start_time = time()
        shape_3d_features = get_shape3d_features(mask, voxel_spacing; verbose=verbose)
        merge!(radiomic_features, shape_3d_features)
        if verbose
            println("3D shape feature extraction time = $(time() - shape_3d_start_time) sec")
            print_features("3D Shape Features", shape_3d_features)
        end

        # GLSZM features
        glszm_start_time = time()
        glszm_features = get_glszm_features(img, mask, voxel_spacing; 
                                      n_bins=n_bins, 
                                      bin_width=bin_width, 
                                      verbose=verbose)
        merge!(radiomic_features, glszm_features)
        if verbose
            println("GLSZM feature extraction time = $(time() - glszm_start_time) sec")
            print_features("GLSZM Features", glszm_features)
        end

        # NGTDM features
        ngtdm_start_time = time()
        ngtdm_features = get_ngtdm_features(img, mask, voxel_spacing; 
                                      n_bins=n_bins, 
                                      bin_width=bin_width, 
                                      verbose=verbose)
        merge!(radiomic_features, ngtdm_features)
        if verbose
            println("NGTDM feature extraction time = $(time() - ngtdm_start_time) sec")
            print_features("NGTDM Features", ngtdm_features)
        end

        # GLRLM features
        glrlm_start_time = time()
        glrlm_features = get_glrlm_features(img, mask, voxel_spacing; 
                                      n_bins=n_bins, 
                                      bin_width=bin_width, 
                                      verbose=verbose)
        merge!(radiomic_features, glrlm_features)
        if verbose
            println("GLRLM feature extraction time = $(time() - glrlm_start_time) sec")
            print_features("GLRLM Features", glrlm_features)
        end

        # GLDM features
        gldm_start_time = time()
        gldm_features = get_gldm_features(img, mask, voxel_spacing; 
                                      n_bins=n_bins, 
                                      bin_width=bin_width, 
                                      verbose=verbose)
        merge!(radiomic_features, gldm_features)
        if verbose
            println("GLDM feature extraction time = $(time() - gldm_start_time) sec")
            print_features("GLDM Features", gldm_features)
        end
    end

    if verbose
        println("\n======================")
        println("Total features extracted: $(length(radiomic_features))")
        println("======================\n")
    end

    return radiomic_features
end

"""
Examples of usage:
# 1) Whith specific n_bins: radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; n_bins=64, verbose=true);

# 2) Whith specific bin_width: radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; bin_width=25.0f0, verbose=true);

# 3) Whith default setting: radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=true); use a default bin_width (25)
"""

end 


