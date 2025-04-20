module Radiomics

include("first_order_features.jl")

function extract_features(img, mask)
    println("Extracting...")

    first_order_features(img, mask)
end

end


