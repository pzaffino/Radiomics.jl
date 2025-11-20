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

function extract_radiomic_features(img_input, mask_input, voxel_spacing_input; force_2d::Bool=false, force_2d_dimension::Int=1, verbose::Bool=false)::Dict{String, Float32}
    """
    Extracts radiomic features from the given image and mask inputs. 
    In case of two-dimensional input, 2D features will be extracted and returned for the single slice.

    # Arguments
    - img_input: The input image from which radiomic features will be extracted. Can be 2D or 3D.
    - mask_input: The mask image defining the region of interest for feature extraction. Can be 2D or 3D.
    - voxel_spacing_input: A tuple or array specifying the voxel spacing of the input image. Can be bidimensional or three-dimensional.
    - force_2d::Bool: If true, forces the extraction to be performed in 2D by removing one dimension from the voxel spacing. This is useful when the input data is three-dimensional but should be treated as two-dimensional.
    - force_2d_dimension::Int: Specifies which dimension to eliminate when force_2d is true. For example, if set to 1, the first dimension will be removed.
    - verbose::Bool: If true, enables verbose output for debugging or detailed processing information.

    # Returns
    A dictionary (Dict{String, Float32}) containing the extracted radiomic features as key-value pairs.
    """
    if verbose
        println("Extracting radiomic features...")
    end

    radiomic_features = Dict{String, Float32}()

    # Cast inputs
    img, mask, voxel_spacing = prepare_inputs(img_input, mask_input, voxel_spacing_input, force_2d, force_2d_dimension)

    # Sanity check on inputs
    sanity_check_start_time = time()
    input_sanity_check(img, mask, verbose)
    if verbose
        println("Sanity check time = $(time() - sanity_check_start_time) sec")
    end

    # Extract features based on dimensionality
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

    # GLCM Features (works for both 2D and 3D)
    glcm_start_time = time()
    glcm_features = get_glcm_features(img, mask, voxel_spacing; verbose=verbose)
    merge!(radiomic_features, glcm_features)
    if verbose
        println("GLCM feature extraction time = $(time() - glcm_start_time) sec")
        print_features("GLCM Features", glcm_features)
    end

    # Shape features
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
        glszm_features = get_glszm_features(img, mask, voxel_spacing, verbose=verbose)
        merge!(radiomic_features, glszm_features)
        if verbose
            println("GLSZM feature extraction time = $(time() - glszm_start_time) sec")
            print_features("GLSZM Features", glszm_features)
        end

        # NGTDM features
        ngtdm_start_time = time()
        ngtdm_features = get_ngtdm_features(img, mask, voxel_spacing, verbose=verbose)
        merge!(radiomic_features, ngtdm_features)
        if verbose
            println("NGTDM feature extraction time = $(time() - ngtdm_start_time) sec")
            print_features("NGTDM Features", ngtdm_features)
        end

        # GLRLM features
        glrlm_start_time = time()
        glrlm_features = get_glrlm_features(img, mask, voxel_spacing, verbose=verbose)
        merge!(radiomic_features, glrlm_features)
        if verbose
            println("GLRLM feature extraction time = $(time() - glrlm_start_time) sec")
            print_features("GLRLM Features", glrlm_features)
        end

        # GLDM features
        gldm_start_time = time()
        gldm_features = get_gldm_features(img, mask, voxel_spacing, verbose=verbose)
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

function print_features(title::String, features::Dict{String, Float32})
    """
    Helper function to print features in a formatted list.
    """
    println("\n--- $title ---")
    sorted_keys = sort(collect(keys(features)))
    for (i, k) in enumerate(sorted_keys)
        println("  $i. $(rpad(k, 35)) => $(features[k])")
    end
    println("Subtotal: $(length(features)) features")
    println("---------------------\n")
end

function prepare_inputs(img_input, mask_input, voxel_spacing_input, force_2d, force_2d_dimension)
    img = Float32.(img_input)
    mask = BitArray(mask_input .!= 0.0f0)
    
    voxel_spacing = Float32.(voxel_spacing_input)

    if ndims(img) == 3 && ndims(mask) == 3 && force_2d
        if force_2d_dimension < 1 || force_2d_dimension > 3
            throw(ArgumentError("force_2d_dimension must be between 1 and 3"))
        end
        voxel_spacing = Float32.(voxel_spacing_input[setdiff(1:3, force_2d_dimension)])
    end

    return img, mask, voxel_spacing
end

function input_sanity_check(img, mask, verbose::Bool)
    if verbose
        println("Running input sanity check...")
    end

    if size(img) != size(mask)
        throw(ArgumentError("img and mask have different size!"))
    end
end

function keep_largest_component(mask::AbstractArray{Bool})
    return mask
end

function pad_mask(mask::AbstractArray, pad::Int)
    sz = size(mask)
    new_shape = ntuple(i -> sz[i] + 2*pad, ndims(mask))
    new_mask = falses(new_shape)
    ranges = ntuple(i -> (1+pad):(sz[i]+pad), ndims(mask))
    new_mask[ranges...] .= mask
    return new_mask
end

end # module Radiomics

# Esegui come: 
# radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=true);