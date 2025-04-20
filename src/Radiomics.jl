module Radiomics

include("first_order_features.jl")

function extract_features(img, mask, verbose=false)
    println("Extracting...")

    # Sanity check on inputs
    input_sanity_check(img, mask, verbose)

    # First order features
    first_order_features(img, mask, verbose)
end


function input_sanity_check(img, mask, verbose)

    if verbose
        println("Running input sanity check...")
    end

    @assert size(img) == size(mask) "img and mask have different size!"
end

end


