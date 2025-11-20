
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

"""
    get_neighbors(idx, dims)

Gets the 26-connected neighbors of a voxel in a 3D image.

# Arguments
- `idx`: The linear index of the voxel.
- `dims`: The dimensions of the image.

# Returns
- A vector of linear indices of the neighbors.
"""
function get_neighbors(idx, dims)
    neighbors = []
    cartesian_idx = CartesianIndices(dims)[idx]

    for dz in -1:1, dy in -1:1, dx in -1:1
        if dz == 0 && dy == 0 && dx == 0
            continue
        end

        new_cartesian_idx = cartesian_idx + CartesianIndex(dx, dy, dz)

        if checkbounds(Bool, CartesianIndices(dims), new_cartesian_idx)
            push!(neighbors, LinearIndices(dims)[new_cartesian_idx])
        end
    end

    return neighbors
end
