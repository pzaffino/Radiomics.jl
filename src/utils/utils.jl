"""
    bounding_box(img::AbstractArray{Float64},
                 mask::AbstractArray,
                 verbose::Bool;
                 log_buffer::Union{Vector{String},Nothing}=nothing)::Tuple{AbstractArray{Float64}, BitArray}

    Computes the bounding box of a binary mask and crops both the image and the mask
    to the smallest region containing all voxels where `mask == 1`.
    Works for both 2D and 3D arrays.

    # Arguments
    - `img`: The input image (2D or 3D array).
    - `mask`: The binary mask defining the region of interest (same shape as `img`).
    - `verbose`: If `true`, prints the original and cropped sizes with reduction percentage.

    # Returns
    - `cropped_img`: image array cropped to the bounding box of the mask.
    - `cropped_mask`: binary mask array (`BitArray`) cropped to the same bounding box."""
function bounding_box(img::AbstractArray{Float64},
                       mask::AbstractArray,
                       verbose::Bool;
                       log_buffer::Union{Vector{String},Nothing}=nothing)::Tuple{AbstractArray{Float64}, BitArray}
    function _bb_log(msg)
        if !isnothing(log_buffer)
            push!(log_buffer, msg)
        else
            println(msg)
        end
    end

    if verbose
        _bb_log("Calculating the bounding box")
        _bb_log("Image data: $(size(img)) | $(prod(size(img))) voxels")
        _bb_log("Mask data: $(size(mask)) | $(count(mask .== 1)) voxels")
    end

    idx = findall(mask .== 1)
    n = ndims(mask)

    mins = [typemax(Int) for _ in 1:n]
    maxs = [typemin(Int) for _ in 1:n]

    for i in idx
        for d in 1:n
            if i[d] < mins[d]; mins[d] = i[d]; end
            if i[d] > maxs[d]; maxs[d] = i[d]; end
        end
    end

    ranges = [mins[d]:maxs[d] for d in 1:n]

    cropped_img  = img[ranges...]
    cropped_mask = mask[ranges...]

    if verbose
        img_size = size(cropped_img)
        ct_voxels = prod(img_size)
        image_crop_perc = 100 - prod(img_size) / prod(size(img)) * 100
        _bb_log("Cropped image data: $img_size | $ct_voxels voxels | $(round(image_crop_perc, digits=2))% reduction ")
    end

    return cropped_img, BitArray(cropped_mask)
end

"""
    label_components(mask::AbstractArray{Bool})::Array{Int}

    Labels connected components in a binary mask using 26-connectivity for 3D
    (or 8-connectivity for 2D).

    # Arguments:
        - `mask`: The input binary mask.

    # Returns:
        - An array of the same size as `mask` with integer labels.
    """
function label_components(mask::AbstractArray{Bool})::Array{Int}
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
    function discretize_image(img::AbstractArray{Float64},
                               mask::BitArray;
                               n_bins::Union{Int,Nothing}=nothing,
                               bin_width::Union{Float64,Nothing}=nothing,
                               vmin::Union{Float64,Nothing}=nothing,
                               vmax::Union{Float64,Nothing}=nothing)::Tuple{Array{Int}, Int, Vector{Int}, Float64}

    Discretizes the input image for radiomics feature calculation. 
    Takes into account only the voxels within the provided mask.

    You can specify EITHER n_bins (number of bins) OR bin_width (width of each bin), but not both.
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins is calculated automatically
    - If neither is specified, defaults to bin_width=25.0

    This function is compatible with all radiomics features: GLCM, GLDM, GLRLM, GLSZM, NGTDM, etc.

    # Arguments:
        - `img`: The input 3D image as Float64 array.
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
        disc, n_bins, levels, bw = discretize_image(img, mask, bin_width=25.0)
        
        # Default (bin_width=25.0)
        disc, n_bins, levels, bw = discretize_image(img, mask)
    """
function discretize_image(img::AbstractArray{Float64},
    mask::BitArray;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    vmin::Union{Float64,Nothing}=nothing,
    vmax::Union{Float64,Nothing}=nothing)::Tuple{Array{Int}, Int, Vector{Int}, Float64}

    if sum(mask) == 0
        return zeros(Int, size(img)), 0, Int[], 0.0
    end

    if isnothing(vmin) || isnothing(vmax)
        vals = view(img, mask)
        vmin = minimum(vals)
        vmax = maximum(vals)
    end

    disc = zeros(Int, size(img))

    if !isnothing(n_bins) && !isnothing(bin_width)
        error("Specify either n_bins or bin_width, not both.")
    elseif isnothing(n_bins) && isnothing(bin_width)
        bin_width = 25.0
    end

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float64(n_bins)
        if bin_width_used ≈ 0.0
            bin_width_used = 1.0
        end
        inv_bin_width = 1.0 / bin_width_used
        @inbounds for i in eachindex(mask)
            if mask[i]
                v = img[i]
                b = min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
                disc[i] = b
            end
        end
    else
        bin_width_used = bin_width
        inv_bin_width = 1.0 / bin_width_used
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
    keep_largest_component(mask::AbstractArray{Bool})::Tuple{AbstractArray{Bool}, Int}

    Keeps only the largest connected component in the binary mask. All other components are removed. 
    If multiple islands are detected, a warning is issued.

    # Arguments:
        - `mask`: The input binary mask (Array).
    
    # Returns:
        - The mask containing only the largest connected component (Array).
    """
function keep_largest_component(mask::AbstractArray{Bool})::Tuple{AbstractArray{Bool}, Int}
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
        @warn "Detected $num_islands separate islands in the mask. Shape features will be computed only on the largest island. First order and texture features will consider all islands."

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
    pad_mask(mask::AbstractArray{Bool}, pad::Int)::BitArray
    
    Pads the input mask with a specified number of layers of false values.
    
    # Arguments:
        - `mask`: The input mask (Array).
        - `pad`: The number of layers to pad around the mask.
    
    # Returns:
        - The padded mask (Array).
    """
@inline function pad_mask(mask::AbstractArray{Bool}, pad::Int)::BitArray
    sz = size(mask)
    new_shape = ntuple(i -> sz[i] + 2 * pad, ndims(mask))
    new_mask = falses(new_shape)
    ranges = ntuple(i -> (1+pad):(sz[i]+pad), ndims(mask))
    new_mask[ranges...] .= mask
    return new_mask
end

"""
    Helper function to print features in a formatted list.
        # Parameters:
        - `title`: The title to display before the features.
        - `features`: A dictionary of features to print.
        # Returns:
        - Nothing. Prints the features to the console."""
function print_features(title::String,
                         features::Dict{String,Any};
                         log_buffer::Union{Vector{String},Nothing}=nothing)::Nothing
    output = String[]
    
    push!(output, "\n--- $title ---")
    
    base_keys = sort([k for k in keys(features) 
                       if !endswith(k, "_std") && !endswith(k, "_min") && !endswith(k, "_max")])
    
    for (i, k) in enumerate(base_keys)
        val = features[k]
        std_key = k * "_std"
        min_key = k * "_min"
        max_key = k * "_max"
        
        padded_key = rpad(k, 42)
        
        if haskey(features, std_key)
            std_val = features[std_key]
            min_val = features[min_key]
            max_val = features[max_key]
            push!(output, "  $i. $(padded_key) => $val (std = $std_val, min = $min_val, max = $max_val)")
        else
            push!(output, "  $i. $(padded_key) => $val")
        end
    end
    
    push!(output, "Subtotal: $(length(base_keys)) features")
    push!(output, "---------------------\n")
    
    if isnothing(log_buffer)
        for line in output
            println(line)
        end
    else
        append!(log_buffer, output)
    end

    return nothing
end

"""
    _cast_inputs(img_input, mask_input, voxel_spacing_input,
                 features, labels, n_bins, bin_width, weighting_norm)

    Converts all input parameters to their correct concrete types.
    Acts as a type barrier before the main feature extraction logic.

    # Returns
    - A NamedTuple with all parameters converted to their correct types.
"""
function _cast_inputs(
    img_input,
    mask_input,
    voxel_spacing_input,
    features,
    labels,
    n_bins,
    bin_width,
    weighting_norm,
    features_std,
    slices_2d,          
    keep_largest_only,  
    get_raw_matrices,   
    verbose             
    )::NamedTuple{
        (:img, :mask, :spacing, :features, :labels, :n_bins, :bin_width, :weighting_norm, :features_std, :slices_2d, :keep_largest_only, :get_raw_matrices, :verbose),
        Tuple{
            Union{Array{Float64,2}, Array{Float64,3}},
            Union{Array{Int,2}, Array{Int,3}},
            Vector{Float64},
            Vector{Symbol},
            Union{Int, Vector{Int}},
            Union{Nothing, Int},
            Union{Nothing, Float64},
            Union{Nothing, String}, 
            Bool,
            Union{Nothing, Vector{Tuple{Int,Int}}},
            Bool,                                     
            Bool,                                     
            Bool                                      
        }
    }

    # --- Perpare input ---
    #For img
    # 2D input: convert to 2D Float64 array
    img::Union{Array{Float64,2}, Array{Float64,3}} = if ndims(img_input) == 2
        convert(Array{Float64,2}, img_input)
    # 3D input: convert to 3D Float64 array
    elseif ndims(img_input) == 3
        convert(Array{Float64,3}, img_input)
    else
        throw(ArgumentError("Input image must be 2D or 3D, got $(ndims(img_input))D"))
    end

    # For Mask
    # 2D input: convert to 2D Float64 array
    mask::Union{Array{Int,2}, Array{Int,3}} = if ndims(mask_input) == 2
        convert(Array{Int,2}, mask_input)
    # 3D input: convert to 3D Float64 array
    elseif ndims(mask_input) == 3
        convert(Array{Int,3}, mask_input)
    else
        throw(ArgumentError("Input mask must be 2D or 3D, got $(ndims(mask_input))D"))
    end

    #Conversion of mask & img and control of Sanity Check
    if verbose
        println("Running Conversion completed. Starting Sanity check...")
    end

    #Sanity Check
    if size(img) != size(mask)
         throw(ArgumentError("img and mask have different size!"))
    end

    # Convert spacing (supports any numeric vector/list)
    spacing::Vector{Float64} = Float64[Float64(s) for s in voxel_spacing_input]
    if length(spacing) > 3
        throw(ArgumentError("voxel_spacing_input is too long! It should have 2 or 3 elements."))
    elseif length(spacing) < 2
        throw(ArgumentError("voxel_spacing_input is too short! It should have 2 or 3 elements."))
    end

    # Convert features (supports String, Vector{String}, Symbol, Vector{Symbol})
    features_out::Vector{Symbol} = if features isa String
        lowercase(features) == "all" ? Symbol[] : [Symbol(lowercase(features))]
    elseif features isa AbstractVector && !isempty(features) && eltype(features) <: AbstractString
        Symbol[Symbol(lowercase(string(f))) for f in features]
    elseif features isa AbstractVector && !isempty(features) && !(eltype(features) <: Symbol)
        Symbol[Symbol(lowercase(string(f))) for f in features]
    else
        Vector{Symbol}(features)
    end

    # Convert labels (supports Int, Vector{Int})
    labels_out::Union{Int,Vector{Int}} = if labels isa AbstractVector
        Int[Int(l) for l in labels]
    elseif !isnothing(labels) && !(labels isa Int)
        Int(labels)
    elseif isnothing(labels)
        Int(1)
    else
        Int(labels)
    end

    # Convert n_bins 
    n_bins_out::Union{Nothing,Int} = isnothing(n_bins) ? nothing : Int(n_bins)

    # Convert bin_width
    bin_width_out::Union{Nothing,Float64} = isnothing(bin_width) ? nothing : Float64(bin_width)

    # Convert weighting_norm
    weighting_norm_out::Union{Nothing,String} = isnothing(weighting_norm) ? nothing : String(weighting_norm)

    #Convert slices_2d
    slices_2d_out::Union{Nothing, Vector{Tuple{Int,Int}}} = if isnothing(slices_2d)
        nothing
    else
        Tuple{Int,Int}[(Int(s[1]), Int(s[2])) for s in slices_2d]
    end

    return (
        img            = img,
        mask           = mask,
        spacing        = spacing,
        features       = features_out,
        labels         = labels_out,
        n_bins         = n_bins_out,
        bin_width      = bin_width_out,
        weighting_norm = weighting_norm_out,
        features_std   = Bool(features_std),
        slices_2d      = slices_2d_out,
        keep_largest_only = Bool(keep_largest_only),
        get_raw_matrices  = Bool(get_raw_matrices),
        verbose           = Bool(verbose)
    )
end

"""
    Validates the binning parameters for feature extraction.
        # Parameters:
        - `img_input`: The input image (Array).
        - `mask_input`: The mask defining the region of interest (Array).
        - `bin_width`: The width of each bin (Float64).
        # Returns:
        - Nothing. Prints a warning if the binning parameters are invalid."""
function validate_binning_parameters(img_input::AbstractArray{Float64},
                                      mask_input::BitArray,
                                      bin_width::Float64)::Nothing

    # Calculate range and estimated number of bins
    roi_vals = img_input[mask_input]
    val_min = minimum(roi_vals)
    val_max = maximum(roi_vals)
    val_range = val_max - val_min
    estimated_bins = Int(ceil(val_range / bin_width))

    if estimated_bins < 3
        @warn """
        With the current bin_width ($bin_width), the current image range ($val_min, $val_max) is included in only $estimated_bins bin(s).
        This may produce unreliable features.
        Consider specifying a smaller bin_width (or using n_bins instead) 
        or to properly scale the image intensity range.
        """
    end
end

"""
    Extracts and checks the mask for a specific label.
        # Parameters:
        - `mask_input`: The mask defining the region of interest (Array).
        - `label`: The label to extract from the mask (Int).
        # Returns:
        - A tuple containing the extracted mask and the voxel count."""
function extract_and_check_mask(mask_input::AbstractArray{Int},
                                 label::Int)::Tuple{BitArray, Int}
    mask_to_use = (mask_input .== label)
    # Check if label exists
    voxel_count = sum(mask_to_use)
    if voxel_count == 0
        @warn "Label $label not found in mask (no voxels with this value). Skipping."
    end
    
    return mask_to_use, voxel_count

end