module Radiomics

include("first_order_features.jl")


function extract_radiomic_features(img, mask; binarize_mask=false, verbose=false)
    println("Extracting...")

    radiomic_features = Dict()

    # Sanity check on inputs
    sanity_check_start_time = time()
    input_sanity_check(img, mask, verbose)
    if verbose
        println("Sanity check time = $(time() - sanity_check_start_time)")
    end

    # Binarize mask
    if binarize_mask == true
        binarize_start_time = time()
        mask = run_mask_binarization(mask)
        if verbose
            println("Masking time = $(time() - binarize_start_time)")
        end
    end

    # First order features
    first_order_start_time = time()
    first_order_features = get_first_order_features(img, mask, verbose)
    merge!(radiomic_features, first_order_features)
    if verbose
        println("First order time = $(time() - first_order_start_time)")
    end

    return radiomic_features

end


function input_sanity_check(img, mask, verbose)

    if verbose
        println("Running input sanity check...")
    end

    img_offset = [img.header.qoffset_x, img.header.qoffset_y, img.header.qoffset_z]
    mask_offset = [mask.header.qoffset_x, mask.header.qoffset_y, mask.header.qoffset_z]

    @assert size(img) == size(mask) "img and mask have different size!"
    @assert img.header.pixdim[2:4] == mask.header.pixdim[2:4] "img and mask have different voxel size!"
    @assert img_offset == mask_offset "img and mask have different offset!"

end


function run_mask_binarization(mask)

    mask = mask .- minimum(mask)
    mask = mask ./ maximum(mask)

    return mask

end


end

