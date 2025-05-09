module Radiomics

include("first_order_features.jl")

function extract_radiomic_features(img_input,
                                   mask_input,
                                   voxel_spacing_input;
                                   verbose::Bool=false)::Dict{String, Float32}

    println("Extracting...")

    radiomic_features = Dict{String, Float32}()

    # Cast inputs
    img::Array{Float32, 3} = Float32.(img_input)
    mask::BitArray{3} = BitArray(mask_input .!= 0.0f0)
    voxel_spacing::Vector{Float32} = Float32.(voxel_spacing_input)

    # Sanity check on inputs
    sanity_check_start_time::Float64 = time()
    input_sanity_check(img, mask, verbose)
    if verbose
        println("Sanity check time = $(time() - sanity_check_start_time)")
    end

    # First order features
    first_order_start_time::Float64 = time()
    first_order_features::Dict{String, Float32} = get_first_order_features(img, mask, voxel_spacing, verbose)
    merge!(radiomic_features, first_order_features)
    if verbose
        println("First order time = $(time() - first_order_start_time)")
    end

    return radiomic_features

end


function input_sanity_check(img, mask, verbose::Bool)

    if verbose
        println("Running input sanity check...")
    end

    @assert size(img) == size(mask) "img and mask have different size!"

end


end

