using Images

"""
    discretize_image(img::Array{Float32,3}, mask::BitArray{3}; 
                        n_bins::Union{Int,Nothing}=nothing,
                        bin_width::Union{Float32,Nothing}=nothing)

    Discretizes the input image for radiomics feature calculation. 
    Takes into account only the voxels within the provided mask.

    You can specify EITHER n_bins (number of bins) OR bin_width (width of each bin), but not both.
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins is calculated automatically
    - If neither is specified, defaults to bin_width=25.0f0

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
        
        # Default (bin_width=25.0f0)
        disc, n_bins, levels, bw = discretize_image(img, mask)
    """
function discretize_image(img::Array{Float32,3},
                         mask::BitArray{3};
                         n_bins::Union{Int,Nothing}=nothing,
                         bin_width::Union{Float32,Nothing}=nothing)
    
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
        bin_width = 25.0f0
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

"""
    keep_largest_component(mask::AbstractArray{Bool})

    Keeps only the largest connected component in the binary mask. All other components are removed. 
    If multiple islands are detected, a warning is issued.

    # Arguments:
        - `mask`: The input binary mask (Array).
    
    # Returns:
        - The mask containing only the largest connected component (Array).
    """
function keep_largest_component(mask::AbstractArray{Bool})
    if sum(mask) == 0
        return mask
    end
    
    # Label connected components
    labels = label_components(mask)
    
    # Count voxels in each component (excluding background=0)
    component_sizes = Dict{Int, Int}()
    for label_val in labels
        if label_val > 0
            component_sizes[label_val] = get(component_sizes, label_val, 0) + 1
        end
    end
    
    if isempty(component_sizes)
        return mask
    end
    
    # Check if there are multiple islands
    num_islands = length(component_sizes)
    if num_islands > 1
        @warn "Detected $num_islands separate islands in the mask. 3D features will be computed only on the largest island. First order and texture features will consider all islands."
        
        # Optional: print sizes of all islands for debugging
        sorted_sizes = sort(collect(component_sizes), by=x->x[2], rev=true)
        println("Island sizes (in voxels):")
        for (i, (label_val, size)) in enumerate(sorted_sizes)
            println("  Island $i: $size voxels")
        end
    end
    
    # Find largest component
    largest_label = argmax(component_sizes)
    
    # Create new mask with only largest component
    largest_component_mask = labels .== largest_label
    
    return largest_component_mask
end

"""
    pad_mask(mask::AbstractArray, pad::Int)
    
    Pads the input mask with a specified number of layers of false values.
    
    # Arguments:
        - `mask`: The input mask (Array).
        - `pad`: The number of layers to pad around the mask.
    
    # Returns:
        - The padded mask (Array).
    """
function pad_mask(mask::AbstractArray, pad::Int)
    sz = size(mask)
    new_shape = ntuple(i -> sz[i] + 2*pad, ndims(mask))
    new_mask = falses(new_shape)
    ranges = ntuple(i -> (1+pad):(sz[i]+pad), ndims(mask))
    new_mask[ranges...] .= mask
    return new_mask
end

"""
    Performs sanity checks on the input image and mask.
        # Parameters:
        - `img`: The input image (Array).
        - `mask`: The mask defining the region of interest (Array).
        - `verbose`: If true, prints progress messages.
        # Returns:
        - Nothing. Throws an error if the inputs are invalid."""
function input_sanity_check(img, mask, verbose::Bool)
    if verbose
        println("Running input sanity check...")
    end

    if size(img) != size(mask)
        throw(ArgumentError("img and mask have different size!"))
    end
end

"""
    Helper function to print features in a formatted list.
        # Parameters:
        - `title`: The title to display before the features.
        - `features`: A dictionary of features to print.
        # Returns:
        - Nothing. Prints the features to the console.
    """
function print_features(title::String, features::Dict{String, Float32})
    println("\n--- $title ---")
    sorted_keys = sort(collect(keys(features)))
    for (i, k) in enumerate(sorted_keys)
        println("  $i. $(rpad(k, 35)) => $(features[k])")
    end
    println("Subtotal: $(length(features)) features")
    println("---------------------\n")
end

"""
    Prepares and validates the input image, mask, and voxel spacing.
        # Parameters:
        - `img_input`: The input image (Array).
        - `mask_input`: The mask defining the region of interest (Array).
        - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
        - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
        - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
        # Returns:
        - A tuple containing the prepared image, mask, and voxel spacing."""
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
