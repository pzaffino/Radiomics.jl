function discretize_image(img::Array{Float32,3},
                         mask::BitArray{3};
                         n_bins::Union{Int,Nothing}=nothing,
                         bin_width::Union{Float32,Nothing}=nothing)
    """
    discretize_image(img::Array{Float32,3}, mask::BitArray{3}; 
                        n_bins::Union{Int,Nothing}=nothing,
                        bin_width::Union{Float32,Nothing}=nothing)

    Discretizes the input image for radiomics feature calculation. 
    Takes into account only the voxels within the provided mask.

    You can specify EITHER n_bins (number of bins) OR bin_width (width of each bin), but not both.
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins is calculated automatically
    - If neither is specified, defaults to n_bins=32

    This function is compatible with all radiomics features: GLCM, GLDM, GLRLM, GLSZM, NGTDM, etc.

    # Arguments:
        - `img`: The input 3D image as Float32 array.
        - `mask`: The mask defining the region of interest as BitArray.
        - `n_bins`: The number of discrete gray levels (optional).
        - `bin_width`: The width of each bin (optional).

    # Returns:
        - `disc`: The discretized image as an array of integers.
        - `n_bins_actual`: The actual number of discrete gray levels present.
        - `gray_levels`: A vector containing the unique gray levels present in the ROI.
        - `bin_width_used`: The bin width used for discretization.

    # Examples:
        # Using fixed number of bins (bin_width calculated automatically)
        disc, n_bins, levels, bw = discretize_image(img, mask, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        disc, n_bins, levels, bw = discretize_image(img, mask, bin_width=25.0f0)
        
        # Default (32 bins)
        disc, n_bins, levels, bw = discretize_image(img, mask)
    """
    masked_indices = findall(mask)
    if isempty(masked_indices)
        return zeros(Int, size(img)), 0, Int[], 0.0f0
    end

    vals = img[mask]
    vmin = minimum(vals)
    vmax = maximum(vals)


    if !isnothing(n_bins) && !isnothing(bin_width)
        error("Specify either n_bins or bin_width, not both.")
        
    elseif isnothing(n_bins) && isnothing(bin_width)
        bin_width=25.0f0
    end

    disc = zeros(Int, size(img))

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float32(n_bins)
        if bin_width_used ≈ 0.0f0
            bin_width_used = 1.0f0
        end
        
        @inbounds for idx in masked_indices
            v = img[idx]
            b = min(Int(floor((v - vmin) / bin_width_used)) + 1, n_bins)
            disc[idx] = b
        end
    else

        bin_width_used = bin_width
        bin_offset = Int(floor(vmin / bin_width_used))
        
        @inbounds for idx in masked_indices
            v = img[idx]
            b = Int(floor(v / bin_width_used)) - bin_offset + 1
            disc[idx] = b
        end
    end

    gray_levels = sort(unique(disc[mask]))
    n_bins_actual = length(gray_levels)

    return disc, n_bins_actual, gray_levels, bin_width_used
end

function get_neighbors(idx, dims)
    """
    get_neighbors(idx, dims)

    Gets the 26-connected neighbors of a voxel in a 3D image.

    # Arguments
    - `idx`: The linear index of the voxel.
    - `dims`: The dimensions of the image.

    # Returns
    - A vector of linear indices of the neighbors.
    """
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


function keep_largest_component(mask::AbstractArray{Bool})
    """
    Keeps only the largest connected component in the binary mask.
        # Parameters:
        - `mask`: The input binary mask (Array).
        # Returns:
        - The mask containing only the largest connected component (Array).
    """
    return mask
end


function pad_mask(mask::AbstractArray, pad::Int)
    """
    Pads the input mask with a specified number of layers of false values.
        # Parameters:
        - `mask`: The input mask (Array).
        - `pad`: The number of layers to pad around the mask.
        # Returns:
        - The padded mask (Array)."""
    sz = size(mask)
    new_shape = ntuple(i -> sz[i] + 2*pad, ndims(mask))
    new_mask = falses(new_shape)
    ranges = ntuple(i -> (1+pad):(sz[i]+pad), ndims(mask))
    new_mask[ranges...] .= mask
    return new_mask
end

