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
function get_ngtdm_features(img, mask, voxel_spacing; 
                           n_bins::Union{Int,Nothing}=nothing,
                           bin_width::Union{Float32,Nothing}=nothing,
                           verbose=false)
    if verbose
        if !isnothing(n_bins)
            println("NGTDM calculation with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("NGTDM calculation with bin_width=$(bin_width)...")
        else
            println("NGTDM calculation with 32 bins (default)...")
        end
    end

    ngtdm_features = Dict{String, Float32}()

    # 1. Discretize the image
    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective gray level: $(n_bins_actual)")
    end

    # 2. Calculate the NGTDM matrix
    P_ngtdm, gray_levels = calculate_ngtdm_matrix(discretized_img, mask, verbose)

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
function calculate_ngtdm_matrix(discretized_img, mask, verbose)
    if verbose
        println("Calculating NGTDM matrix...")
    end

    masked_img = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)
    gl_map = Dict(gl => i for (i, gl) in enumerate(gray_levels))

    P_ngtdm = zeros(Float32, num_gl, 3)

    for i in eachindex(discretized_img)
        if mask[i]
            gl = discretized_img[i]
            gl_idx = gl_map[gl]

            neighborhood_sum = 0
            neighborhood_count = 0

            for neighbor_idx in get_neighbors(i, size(discretized_img))
                if mask[neighbor_idx]
                    neighborhood_sum += discretized_img[neighbor_idx]
                    neighborhood_count += 1
                end
            end

            if neighborhood_count > 0
                neighborhood_avg = neighborhood_sum / neighborhood_count
                P_ngtdm[gl_idx, 1] += 1
                P_ngtdm[gl_idx, 2] += abs(gl - neighborhood_avg)
                P_ngtdm[gl_idx, 3] = gl
            end
        end
    end

    return P_ngtdm, gray_levels
end

"""
    calculate_ngtdm_coefficients(P_ngtdm, gray_levels)

    Calculates the coefficients used in the NGTDM feature calculations.

    # Arguments
    - `P_ngtdm`: The NGTDM matrix.
    - `gray_levels`: The gray levels present in the ROI.

    # Returns
    - A tuple containing the number of voxels with a valid region, the gray level probability vector, the sum of absolute differences vector, the gray level vector, and the number of gray levels for which `p_i` > 0.
    """
function calculate_ngtdm_coefficients(P_ngtdm, gray_levels)
    Nvp = sum(P_ngtdm[:, 1])
    p_i = P_ngtdm[:, 1] ./ Nvp
    s_i = P_ngtdm[:, 2]
    ivector = Float32.(gray_levels)
    Ngp = sum(P_ngtdm[:, 1] .> 0)

    return Nvp, p_i, s_i, ivector, Ngp
end

"""
    coarseness(p_i, s_i)
    Calculates the Coarseness feature.
    """
function coarseness(p_i, s_i)
    sum_coarse = sum(p_i .* s_i)
    return sum_coarse == 0 ? 1e6 : 1 / sum_coarse
end

"""
    contrast(p_i, s_i, i, Ngp, Nvp)
    Calculates the Contrast feature.
    """
function contrast(p_i, s_i, i, Ngp, Nvp)
    if Ngp <= 1
        return 0.0
    end

    div = Ngp * (Ngp - 1)

    contrast_val = (sum(p_i' .* p_i .* (i' .- i) .^ 2) * sum(s_i) / Nvp)

    return contrast_val / div
end

"""
    busyness(p_i, s_i, i)
    Calculates the Busyness feature.
    """
function busyness(p_i, s_i, i)
    p_zero_mask = p_i .== 0
    i_pi = i .* p_i

    abs_diff = abs.(i_pi' .- i_pi)
    abs_diff[p_zero_mask, :] .= 0
    abs_diff[:, p_zero_mask] .= 0

    sum_abs_diff = sum(abs_diff)

    if sum_abs_diff == 0
        return 0.0
    end

    return sum(p_i .* s_i) / sum_abs_diff
end

"""
    complexity(p_i, s_i, i, Nvp)
    Calculates the Complexity feature.
    """
function complexity(p_i, s_i, i, Nvp)
    p_zero_mask = p_i .== 0
    pi_si = p_i .* s_i

    numerator = pi_si' .+ pi_si
    numerator[p_zero_mask, :] .= 0
    numerator[:, p_zero_mask] .= 0

    divisor = p_i' .+ p_i
    divisor[divisor .== 0] .= 1 # Avoid division by zero

    complexity_val = sum(abs.(i' .- i) .* numerator ./ divisor) / Nvp
    return complexity_val
end

"""
    strength(p_i, s_i, i)
    Calculates the Strength feature.
    """
function strength(p_i, s_i, i)
    sum_s_i = sum(s_i)
    if sum_s_i == 0
        return 0.0
    end

    p_zero_mask = p_i .== 0
    strength_val = (p_i' .+ p_i) .* (i' .- i) .^ 2
    strength_val[p_zero_mask, :] .= 0
    strength_val[:, p_zero_mask] .= 0

    return sum(strength_val) / sum_s_i
end