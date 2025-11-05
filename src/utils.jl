
"""
    discretize(img, mask; bin_width=25)

Discretizes the input image using a fixed bin width, following the pyradiomics implementation.

# Arguments
- `img`: The input image.
- `mask`: The mask defining the region of interest.
- `bin_width`: The width of each bin.

# Returns
- The discretized image.
"""
function discretize(img, mask; bin_width=25)
    masked_img = img[mask]
    min_val = minimum(masked_img)

    discretized = floor.(img ./ bin_width) .- floor.(min_val / bin_width) .+ 1
    discretized[.!mask] .= 0

    return Int.(discretized)
end
