module Radiomics

include("first_order_features.jl")

function extract_features(img, mask, verbose=false)
    println("Extracting...")

    first_order_features(img, mask, verbose)
end

end


