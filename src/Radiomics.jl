module Radiomics
using IterTools: unzip

include("first_order_features.jl")
include("shape2D.jl")
include("shape3D.jl")

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
        println("Extracting...")
    end

    radiomic_features = Dict{String, Float32}()

    # Cast inputs
    img, mask, voxel_spacing = prepare_inputs(img_input, mask_input, voxel_spacing_input, force_2d, force_2d_dimension)

    # Sanity check on inputs
    sanity_check_start_time::Float64 = time()
    input_sanity_check(img, mask, verbose)
    if verbose
        println("Sanity check time = $(time() - sanity_check_start_time)")
    end

    # Extract features
    if ndims(img) == 3
        # First order features
        first_order_start_time::Float64 = time()
        first_order_features::Dict{String, Float32} = get_first_order_features(img, mask, voxel_spacing, verbose)
        merge!(radiomic_features, first_order_features)
        if verbose
            println("First order time = $(time() - first_order_start_time)")
        end
    end

    if ndims(mask) == 2
        println("Starting shape2D")
        shape_2d_start_time::Float64 = time()
        shape_2d_features::Dict{String, Float32} = get_shape2d_features(mask, voxel_spacing, verbose)
        merge!(radiomic_features, shape_2d_features)
        println("Finished shape2D in $(time() - shape_2d_start_time) sec")
    elseif ndims(mask) == 3
        println("Starting shape3D")
        shape_3d_start_time::Float64 = time()
        shape_3d_features::Dict{String, Float32} = get_shape3d_features(mask, voxel_spacing; verbose=verbose)
        merge!(radiomic_features, shape_3d_features)
        println("Finished shape3D in $(time() - shape_3d_start_time) sec")
    end

    return radiomic_features

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
    print(ndims(img), ndims(mask), force_2d, force_2d_dimension)
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


end