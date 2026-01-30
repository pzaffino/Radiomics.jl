"""
    label_components(mask::AbstractArray{Bool})

    Labels connected components in a binary mask using 26-connectivity for 3D
    (or 8-connectivity for 2D).

    # Arguments:
        - `mask`: The input binary mask.

    # Returns:
        - An array of the same size as `mask` with integer labels.
    """
function label_components(mask::AbstractArray{Bool})
    labels = zeros(Int, size(mask))
    current_label = 0
    queue = Int[]
    sizehint!(queue, length(mask)) # Pre-allocate somewhat

    dims = size(mask)
    lin_indices = LinearIndices(dims)
    cart_indices = CartesianIndices(dims)

    # 3D offsets for 26-connectivity, or 2D for 8-connectivity
    # We'll generate them dynamically to handle generic dims
    offsets = Int[]
    if ndims(mask) == 3
        for z in -1:1, y in -1:1, x in -1:1
            (x == 0 && y == 0 && z == 0) && continue
            push!(offsets, lin_indices[CartesianIndex(x + 2, y + 2, z + 2)] - lin_indices[CartesianIndex(2, 2, 2)])
        end
    elseif ndims(mask) == 2
        for y in -1:1, x in -1:1
            (x == 0 && y == 0) && continue
            push!(offsets, lin_indices[CartesianIndex(x + 2, y + 2)] - lin_indices[CartesianIndex(2, 2)])
        end
    end
    # Note: The above offset calculation is a bit tricky with edge cases if not careful 
    # about bounds. It's safer to use CartesianIndices for bounds checking or 
    # standard neighbor iteration. 
    # Let's stick to a robust standard BFS with CartesianIndices to ensure correctness 
    # at edges, similar to the existing get_neighbors but optimized for the queue.

    @inbounds for i in eachindex(mask)
        if mask[i] && labels[i] == 0
            current_label += 1
            labels[i] = current_label
            push!(queue, i)

            while !isempty(queue)
                idx = pop!(queue)
                c_idx = cart_indices[idx]

                # Check neighbors
                for iter in CartesianIndices(ntuple(d -> -1:1, ndims(mask)))
                    all(t -> t == 0, Tuple(iter)) && continue

                    n_c_idx = c_idx + iter

                    if checkbounds(Bool, mask, n_c_idx)
                        n_idx = lin_indices[n_c_idx]
                        if mask[n_idx] && labels[n_idx] == 0
                            labels[n_idx] = current_label
                            push!(queue, n_idx)
                        end
                    end
                end
            end
        end
    end

    return labels
end

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

    # Early check for empty mask
    if sum(mask) == 0
        return zeros(Int, size(img)), 0, Int[], 0.0f0
    end

    # Compute min/max from masked values
    vals = view(img, mask)
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

        inv_bin_width = 1.0f0 / bin_width_used
        # Iterate directly using linear indices without allocating index array
        @inbounds for i in eachindex(mask)
            if mask[i]
                v = img[i]
                b = min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
                disc[i] = b
            end
        end
    else
        bin_width_used = bin_width
        inv_bin_width = 1.0f0 / bin_width_used
        bin_offset = Int(floor(vmin * inv_bin_width))

        @inbounds for i in eachindex(mask)
            if mask[i]
                v = img[i]
                b = Int(floor(v * inv_bin_width)) - bin_offset + 1
                disc[i] = b
            end
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
@inline function get_neighbors(idx, dims)
    # Pre-allocate with maximum possible size (26 for 3D)
    neighbors = Vector{Int}(undef, 26)
    count = 0

    cartesian_idx = CartesianIndices(dims)[idx]
    linear_indices = LinearIndices(dims)
    cartesian_range = CartesianIndices(dims)

    @inbounds for dz in -1:1, dy in -1:1, dx in -1:1
        if dz == 0 && dy == 0 && dx == 0
            continue
        end

        new_cartesian_idx = cartesian_idx + CartesianIndex(dx, dy, dz)

        if checkbounds(Bool, cartesian_range, new_cartesian_idx)
            count += 1
            neighbors[count] = linear_indices[new_cartesian_idx]
        end
    end

    # Return only the filled portion
    return resize!(neighbors, count)
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
        return mask, 0  # Restituisce anche 0 isole
    end

    # Label connected components
    labels = label_components(mask)

    # Find max label to determine array size
    max_label = maximum(labels)

    if max_label == 0
        return mask, 0
    end

    # Count voxels in each component using array (much faster than Dict)
    component_sizes = zeros(Int, max_label)
    @inbounds for label_val in labels
        if label_val > 0
            component_sizes[label_val] += 1
        end
    end

    # Check if there are multiple islands
    num_islands = count(>(0), component_sizes)
    if num_islands > 1
        @warn "Detected $num_islands separate islands in the mask. 3D features will be computed only on the largest island. First order and texture features will consider all islands."

        # Optional: print sizes of all islands for debugging
        sorted_pairs = [(i, component_sizes[i]) for i in 1:max_label if component_sizes[i] > 0]
        sort!(sorted_pairs, by=x -> x[2], rev=true)
        println("Island sizes (in voxels):")
        for (i, (label_val, size)) in enumerate(sorted_pairs)
            println("  Island $i: $size voxels")
        end
    end

    # Find largest component
    largest_label = argmax(component_sizes)

    # Create new mask with only largest component
    largest_component_mask = labels .== largest_label

    return largest_component_mask, num_islands
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
@inline function pad_mask(mask::AbstractArray{Bool}, pad::Int)
    sz = size(mask)
    new_shape = ntuple(i -> sz[i] + 2 * pad, ndims(mask))
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
function input_sanity_check(img::AbstractArray, mask::AbstractArray, verbose::Bool)
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
function print_features(title::String, features::Dict{String,Float32})
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
function prepare_inputs(img_input::AbstractArray,
    mask_input::AbstractArray,
    voxel_spacing_input::AbstractVector,
    force_2d::Bool,
    force_2d_dimension::Int)
    # Handle both 2D and 3D inputs
    if ndims(img_input) == 2
        # 2D input: convert to 2D Float32 array
        img = convert(Array{Float32,2}, img_input)
        mask = BitArray(mask_input .!= 0.0f0)
    elseif ndims(img_input) == 3
        # 3D input: convert to 3D Float32 array
        img = convert(Array{Float32,3}, img_input)
        mask = BitArray(mask_input .!= 0.0f0)

        if force_2d
            if force_2d_dimension < 1 || force_2d_dimension > 3
                throw(ArgumentError("force_2d_dimension must be between 1 and 3"))
            end
            # Use view instead of creating new array for spacing selection
            dims_to_keep = setdiff(1:3, force_2d_dimension)
            voxel_spacing = convert(Vector{Float32}, voxel_spacing_input[dims_to_keep])
            return img, mask, voxel_spacing
        end
    else
        throw(ArgumentError("Input image must be 2D or 3D, got $(ndims(img_input))D"))
    end

    voxel_spacing = convert(Vector{Float32}, voxel_spacing_input)
    return img, mask, voxel_spacing
end

function control(img_input, mask_input, bin_width)
    # Determine effective bin_width
    effective_bin_width = isnothing(bin_width) ? 25.0f0 : bin_width

    # Calculate range and estimated number of bins
    roi_vals = img_input[mask_input.!=0]
    val_min = minimum(roi_vals)
    val_max = maximum(roi_vals)
    val_range = val_max - val_min
    estimated_bins = Int(ceil(val_range / effective_bin_width))

    if estimated_bins < 3
        if isnothing(bin_width)
            @warn """
            The default bin_width (25.0), the current image range ($val_min, $val_max) is included in only $estimated_bins bin(s).
            This may produce unreliable features.
            Consider specifying a smaller bin_width (or using n_bins instead) 
            or to properly scale the image intensity range.
            """
        else
            @warn """
            With the current bin_width ($bin_width), the current image range ($val_min, $val_max) is included in only $estimated_bins bin(s).
            This may produce unreliable features.
            Consider specifying a smaller bin_width (or using n_bins instead) 
            or to properly scale the image intensity range.
            """
        end
    end
end
