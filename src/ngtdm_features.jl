using StatsBase

"""
    get_ngtdm_features(img, mask, voxel_spacing; n_bins=nothing, bin_width=nothing, verbose=false)

    Calculates and returns a dictionary of NGTDM (Neighbouring Gray Tone Difference Matrix) features.

    You can specify EITHER n_bins (number of bins) OR bin_width (fixed bin width):
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins depends on the intensity range
    - If neither is specified, defaults to n_bins=32

    # Arguments
    - `img`: The input image.
    - `mask`: The mask defining the region of interest.
    - `voxel_spacing`: The spacing of the voxels in the image.
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `get_raw_matrices`: If true, returns the raw NGTDM matrix.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.

    # Examples
        # Using fixed number of bins (bin_width calculated automatically)
        features = get_ngtdm_features(img, mask, spacing, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        features = get_ngtdm_features(img, mask, spacing, bin_width=25.0f0)
        
        # Default (32 bins)
        features = get_ngtdm_features(img, mask, spacing)
    """
function get_ngtdm_features(img::AbstractArray{Float64},
                             mask::BitArray,
                             voxel_spacing::Vector{Float64};
                             n_bins::Union{Int,Nothing}=nothing,
                             bin_width::Union{Float64,Nothing}=nothing,
                             get_raw_matrices::Bool=false,
                             verbose::Bool=false)::Dict{String,Any}
    if verbose
        if !isnothing(n_bins)
            println("NGTDM calculation with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("NGTDM calculation with bin_width=$(bin_width)...")
        else
            println("NGTDM calculation with 32 bins (default)...")
        end
    end

    ngtdm_features = Dict{String, Any}()

    # 1. Discretize the image
    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective gray level: $(n_bins_actual)")
    end

    # 2. Calculate the NGTDM matrix
    P_ngtdm, gray_levels = calculate_ngtdm_matrix(discretized_img, mask, verbose)

    if get_raw_matrices
        if verbose
            println("=================================")
            println("NGTDM Matrix Dimensions: $(size(P_ngtdm))  →  $(size(P_ngtdm,1)) gray levels × $(size(P_ngtdm,2)) zone sizes")
            println("Number of zones: $(sum(P_ngtdm))")
            println("NGTDM saved in dictionary.")
            println("=================================")
        end
        ngtdm_features["raw_ngtdm_matrix"] = P_ngtdm
        return ngtdm_features
    end

    # 3. Calculate coefficients
    Nvp, p_i, s_i, ivector, Ngp = calculate_ngtdm_coefficients(P_ngtdm, gray_levels)

    # 4. Calculate features
    ngtdm_features["ngtdm_Coarseness"] = coarseness(p_i, s_i)
    ngtdm_features["ngtdm_Contrast"] = contrast(p_i, s_i, ivector, Ngp, Nvp)
    ngtdm_features["ngtdm_Busyness"] = busyness(p_i, s_i, ivector)
    ngtdm_features["ngtdm_Complexity"] = complexity(p_i, s_i, ivector, Nvp)
    ngtdm_features["ngtdm_Strength"] = strength(p_i, s_i, ivector)

    if verbose
        println("completed, extract $(length(ngtdm_features)) features.")
    end

    return ngtdm_features
end

"""
    calculate_ngtdm_matrix(discretized_img, mask, verbose)

    Calculates the Neighbouring Gray Tone Difference Matrix (NGTDM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the NGTDM matrix and the gray levels present in the ROI.
    """
function calculate_ngtdm_matrix(discretized_img::Array{Int},
                                 mask::BitArray,
                                 verbose::Bool)::Tuple{Matrix{Float64}, Vector{Int}}

    verbose && println("Calculating NGTDM matrix...")

    masked_img  = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl      = length(gray_levels)

    # Lookup array invece di Dict
    min_gl = minimum(gray_levels)
    max_gl = maximum(gray_levels)
    gl_map = zeros(Int, max_gl - min_gl + 1)
    @inbounds for (i, gl) in enumerate(gray_levels)
        gl_map[gl - min_gl + 1] = i
    end

    P_ngtdm = zeros(Float64, num_gl, 3)

    n_dims  = ndims(discretized_img)
    offsets = [CartesianIndex(Tuple(o)) for o in Iterators.product((-1:1 for _ in 1:n_dims)...)
               if !all(iszero, o)]

    sz            = size(discretized_img)
    cart_indices  = CartesianIndices(sz)
    linear_indices = LinearIndices(sz)

    mask_indices  = findall(mask)
    interior_mask = eltype(mask_indices)[]
    border_mask   = eltype(mask_indices)[]
    sizehint!(interior_mask, length(mask_indices))
    sizehint!(border_mask,   length(mask_indices))

    for idx in mask_indices
        t = Tuple(idx)
        if all(t[k] > 1 && t[k] < sz[k] for k in 1:n_dims)
            push!(interior_mask, idx)
        else
            push!(border_mask, idx)
        end
    end

    @inbounds for idx in interior_mask
        gl     = discretized_img[idx]
        gl_idx = gl_map[gl - min_gl + 1]

        neighborhood_sum   = 0
        neighborhood_count = 0

        for o in offsets
            nidx = idx + o
            if mask[nidx]
                neighborhood_sum   += discretized_img[nidx]
                neighborhood_count += 1
            end
        end

        if neighborhood_count > 0
            neighborhood_avg    = neighborhood_sum / neighborhood_count
            P_ngtdm[gl_idx, 1] += 1
            P_ngtdm[gl_idx, 2] += abs(gl - neighborhood_avg)
            P_ngtdm[gl_idx, 3]  = gl
        end
    end

    for idx in border_mask
        gl     = discretized_img[idx]
        gl_idx = gl_map[gl - min_gl + 1]

        neighborhood_sum   = 0
        neighborhood_count = 0

        for o in offsets
            nidx = idx + o
            if checkbounds(Bool, discretized_img, nidx) && mask[nidx]
                neighborhood_sum   += discretized_img[nidx]
                neighborhood_count += 1
            end
        end

        if neighborhood_count > 0
            neighborhood_avg    = neighborhood_sum / neighborhood_count
            P_ngtdm[gl_idx, 1] += 1
            P_ngtdm[gl_idx, 2] += abs(gl - neighborhood_avg)
            P_ngtdm[gl_idx, 3]  = gl
        end
    end

    return P_ngtdm, gray_levels
end

"""
    calculate_ngtdm_coefficients(P_ngtdm::Matrix{Float64},
                                 gray_levels::Vector{Int})::Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Int}
    Calculates the coefficients used in the NGTDM feature calculations.

    # Arguments
    - `P_ngtdm`: The NGTDM matrix.
    - `gray_levels`: The gray levels present in the ROI.

    # Returns
    - A tuple containing the number of voxels with a valid region, the gray level probability vector, the sum of absolute differences vector, the gray level vector, and the number of gray levels for which `p_i` > 0.
    """
function calculate_ngtdm_coefficients(P_ngtdm::Matrix{Float64},
                                      gray_levels::Vector{Int})::Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Int}
    Nvp = sum(P_ngtdm[:, 1])
    p_i = P_ngtdm[:, 1] ./ Nvp
    s_i = P_ngtdm[:, 2]
    ivector = Float64.(gray_levels)
    Ngp = sum(P_ngtdm[:, 1] .> 0)

    return Nvp, p_i, s_i, ivector, Ngp
end

"""
    coarseness(p_i::Vector{Float64}, s_i::Vector{Float64})::Float64
    Calculates the Coarseness feature.
    """
function coarseness(p_i::Vector{Float64}, s_i::Vector{Float64})::Float64
    sum_coarse = sum(p_i .* s_i)
    return sum_coarse == 0 ? 1e6 : 1 / sum_coarse
end

"""
    contrast(p_i::Vector{Float64}, s_i::Vector{Float64}, i::Vector{Float64}, Ngp::Int, Nvp::Float64)::Float64
    Calculates the Contrast feature.
    """
function contrast(p_i::Vector{Float64},
                  s_i::Vector{Float64},
                  i::Vector{Float64},
                  Ngp::Int,
                  Nvp::Float64)::Float64
    Ngp <= 1 && return 0.0
    n = length(p_i)
    val = 0.0
    @inbounds for a in 1:n, b in 1:n
        val += p_i[a] * p_i[b] * (i[a] - i[b])^2
    end
    return val * sum(s_i) / Nvp / (Ngp * (Ngp - 1))
end

"""
    busyness(p_i::Vector{Float64}, s_i::Vector{Float64}, i::Vector{Float64})::Float64
    Calculates the Busyness feature.
    """
function busyness(p_i::Vector{Float64},
                  s_i::Vector{Float64},
                  i::Vector{Float64})::Float64
    n = length(p_i)
    num = 0.0
    den = 0.0
    @inbounds for k in 1:n
        num += p_i[k] * s_i[k]
    end
    @inbounds for a in 1:n
        p_i[a] == 0 && continue
        for b in 1:n
            p_i[b] == 0 && continue
            den += abs(i[a] * p_i[a] - i[b] * p_i[b])
        end
    end
    return den == 0 ? 0.0 : num / den
end

"""
    complexity(p_i::Vector{Float64}, s_i::Vector{Float64}, i::Vector{Float64}, Nvp::Float64)::Float64
    Calculates the Complexity feature.
    """
function complexity(p_i::Vector{Float64},
                    s_i::Vector{Float64},
                    i::Vector{Float64},
                    Nvp::Float64)::Float64
    n = length(p_i)
    val = 0.0
    @inbounds for a in 1:n
        p_i[a] == 0 && continue
        for b in 1:n
            p_i[b] == 0 && continue
            denom = p_i[a] + p_i[b]
            val += abs(i[a] - i[b]) * (p_i[a] * s_i[a] + p_i[b] * s_i[b]) / denom
        end
    end
    return val / Nvp
end

"""
    strength(p_i::Vector{Float64}, s_i::Vector{Float64}, i::Vector{Float64})::Float64
    Calculates the Strength feature.
    """
function strength(p_i::Vector{Float64},
                  s_i::Vector{Float64},
                  i::Vector{Float64})::Float64
    sum_s_i = sum(s_i)
    sum_s_i == 0 && return 0.0
    n = length(p_i)
    val = 0.0
    @inbounds for a in 1:n
        p_i[a] == 0 && continue
        for b in 1:n
            p_i[b] == 0 && continue
            val += (p_i[a] + p_i[b]) * (i[a] - i[b])^2
        end
    end
    return val / sum_s_i
end