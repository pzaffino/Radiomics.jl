
using StatsBase

"""
    get_gldm_features(img, mask, voxel_spacing; n_bins=nothing, bin_width=nothing, gldm_a=0, verbose=false)

    Calculates and returns a dictionary of GLDM (Gray Level Dependence Matrix) features.

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
    - `gldm_a`: The alpha parameter for the GLDM calculation.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.

    # Examples
        # Using fixed number of bins (bin_width calculated automatically)
        features = get_gldm_features(img, mask, spacing, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        features = get_gldm_features(img, mask, spacing, bin_width=25.0f0)
        
        # Default (32 bins)
        features = get_gldm_features(img, mask, spacing)
"""
function get_gldm_features(img, mask, voxel_spacing;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    gldm_a=0,
    verbose=false)
    if verbose
        if !isnothing(n_bins)
            println("Calculating GLDM with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("GLDM calculation with bin_width=$(bin_width)...")
        else
            println("GLDM calculation with 32 bins (default)...")
        end
    end

    gldm_features = Dict{String,Float32}()

    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective Gray level utilized: $(n_bins_actual)")
    end

    P_gldm, gray_levels = calculate_gldm_matrix(discretized_img, mask, gldm_a, verbose)

    Nz, pd, pg, ivector, jvector = calculate_gldm_coefficients(P_gldm, gray_levels)

    # Pre-compute matrices for combined features to avoid redundant reshaping
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    i_sq = i_mat .^ 2
    j_sq = j_mat .^ 2

    gldm_features["gldm_SmallDependenceEmphasis"] = gldm_small_dependence_emphasis(pd, jvector, Nz)
    gldm_features["gldm_LargeDependenceEmphasis"] = gldm_large_dependence_emphasis(pd, jvector, Nz)
    gldm_features["gldm_GrayLevelNonUniformity"] = gldm_gray_level_non_uniformity(pg, Nz)
    gldm_features["gldm_DependenceNonUniformity"] = gldm_dependence_non_uniformity(pd, Nz)
    gldm_features["gldm_DependenceNonUniformityNormalized"] = gldm_dependence_non_uniformity_normalized(pd, Nz)
    gldm_features["gldm_GrayLevelVariance"] = gldm_gray_level_variance(pg, ivector, Nz)
    gldm_features["gldm_DependenceVariance"] = gldm_dependence_variance(pd, jvector, Nz)
    gldm_features["gldm_DependenceEntropy"] = gldm_dependence_entropy(P_gldm, Nz)
    gldm_features["gldm_LowGrayLevelEmphasis"] = gldm_low_gray_level_emphasis(pg, ivector, Nz)
    gldm_features["gldm_HighGrayLevelEmphasis"] = gldm_high_gray_level_emphasis(pg, ivector, Nz)
    gldm_features["gldm_SmallDependenceLowGrayLevelEmphasis"] = gldm_small_dependence_low_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    gldm_features["gldm_SmallDependenceHighGrayLevelEmphasis"] = gldm_small_dependence_high_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    gldm_features["gldm_LargeDependenceLowGrayLevelEmphasis"] = gldm_large_dependence_low_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    gldm_features["gldm_LargeDependenceHighGrayLevelEmphasis"] = gldm_large_dependence_high_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)

    if verbose
        println("Completed! Extract $(length(gldm_features)) features.")
    end

    return gldm_features
end

"""
    calculate_gldm_matrix(discretized_img, mask, gldm_a, verbose)

    Calculates the Gray Level Dependence Matrix (GLDM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `gldm_a`: The alpha parameter for the GLDM calculation.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLDM matrix and the gray levels present in the ROI.
"""
function calculate_gldm_matrix(discretized_img, mask, gldm_a, verbose)
    if verbose
        println("Calculating GLDM matrix...")
    end

    masked_img = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)
    gl_map = Dict(gl => i for (i, gl) in enumerate(gray_levels))

    # Max dependence is 26 (neighbors) + 1 (center) = 27 for 3D
    max_dependence = 27
    P_gldm = zeros(Int, num_gl, max_dependence)

    for i in eachindex(discretized_img)
        if mask[i]
            gl = discretized_img[i]
            gl_idx = gl_map[gl]

            dependence_count = 0

            for neighbor_idx in get_neighbors(i, size(discretized_img))
                if mask[neighbor_idx] && abs(gl - discretized_img[neighbor_idx]) <= gldm_a
                    dependence_count += 1
                end
            end

            # Add center voxel (always dependent)
            dependence_count += 1

            P_gldm[gl_idx, dependence_count] += 1
        end
    end

    # Trim empty columns
    last_col = findlast(sum(P_gldm, dims=1) .> 0)
    last_col = last_col === nothing ? 0 : last_col[2]
    P_gldm = P_gldm[:, 1:last_col]

    return P_gldm, gray_levels
end

"""
    calculate_gldm_coefficients(P_gldm, gray_levels)

    Calculates the coefficients used in the GLDM feature calculations.

    # Arguments
    - `P_gldm`: The GLDM matrix.
    - `gray_levels`: The gray levels present in the ROI.

    # Returns
    - A tuple containing the number of zones, sum over sizes, sum over gray levels, gray level vector, and size vector.
"""
function calculate_gldm_coefficients(P_gldm, gray_levels)
    Nz = sum(P_gldm)
    Nz = Nz == 0 ? 1.0f-6 : Float32(Nz)

    pd = vec(sum(P_gldm, dims=1))
    pg = vec(sum(P_gldm, dims=2))
    ivector = Float32.(gray_levels)
    jvector = Float32.(1:size(P_gldm, 2))

    return Nz, pd, pg, ivector, jvector
end

# Feature implementations - optimized for vectorized operations
function gldm_small_dependence_emphasis(pd, jvector, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in eachindex(jvector)
        result += pd[j] / (jvector[j]^2)
    end
    return result * inv_Nz
end

function gldm_large_dependence_emphasis(pd, jvector, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in eachindex(jvector)
        result += pd[j] * (jvector[j]^2)
    end
    return result * inv_Nz
end

function gldm_gray_level_non_uniformity(pg, Nz)
    sum_sq = 0.0f0
    @inbounds for val in pg
        sum_sq += val^2
    end
    return sum_sq / Nz
end

function gldm_dependence_non_uniformity(pd, Nz)
    sum_sq = 0.0f0
    @inbounds for val in pd
        sum_sq += val^2
    end
    return sum_sq / Nz
end

gldm_dependence_non_uniformity_normalized(pd, Nz) = gldm_dependence_non_uniformity(pd, Nz) / Nz

"""
    gldm_gray_level_variance(pg, ivector, Nz)
    Calculates the Gray Level Variance feature.
    # Arguments
    - `pg`: The sum over gray levels vector.
    - `ivector`: The gray level vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Gray Level Variance feature value.
"""
function gldm_gray_level_variance(pg, ivector, Nz)
    inv_Nz = 1.0f0 / Nz
    u_i = 0.0f0
    @inbounds for i in eachindex(ivector)
        u_i += pg[i] * ivector[i]
    end
    u_i *= inv_Nz

    variance = 0.0f0
    @inbounds for i in eachindex(ivector)
        diff = ivector[i] - u_i
        variance += pg[i] * diff * diff
    end
    return variance * inv_Nz
end

"""
    gldm_dependence_variance(pd, jvector, Nz)
    Calculates the Dependence Variance feature.
    # Arguments
    - `pd`: The sum over sizes vector.
    - `jvector`: The size vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Dependence Variance feature value."""
function gldm_dependence_variance(pd, jvector, Nz)
    inv_Nz = 1.0f0 / Nz
    u_j = 0.0f0
    @inbounds for j in eachindex(jvector)
        u_j += pd[j] * jvector[j]
    end
    u_j *= inv_Nz

    variance = 0.0f0
    @inbounds for j in eachindex(jvector)
        diff = jvector[j] - u_j
        variance += pd[j] * diff * diff
    end
    return variance * inv_Nz
end

"""
    gldm_dependence_entropy(P_gldm, Nz)
    Calculates the Dependence Entropy feature.
    # Arguments
    - `P_gldm`: The GLDM matrix.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Dependence Entropy feature value.
    """
function gldm_dependence_entropy(P_gldm, Nz)
    inv_Nz = 1.0f0 / Nz
    entropy = 0.0f0
    @inbounds for val in P_gldm
        if val > 0
            p = val * inv_Nz
            entropy -= p * log2(p)
        end
    end
    return entropy
end

function gldm_low_gray_level_emphasis(pg, ivector, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for i in eachindex(ivector)
        result += pg[i] / (ivector[i]^2)
    end
    return result * inv_Nz
end

function gldm_high_gray_level_emphasis(pg, ivector, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for i in eachindex(ivector)
        result += pg[i] * (ivector[i]^2)
    end
    return result * inv_Nz
end

"""
    gldm_small_dependence_low_gray_level_emphasis(P_gldm, ivector
    , jvector, Nz)
    Calculates the Small Dependence Low Gray Level Emphasis feature.
    # Arguments
    - `P_gldm`: The GLDM matrix.
    - `ivector`: The gray level vector.
    - `jvector`: The size vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Small Dependence Low Gray Level Emphasis feature value."""
function gldm_small_dependence_low_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in axes(P_gldm, 2), i in axes(P_gldm, 1)
        if P_gldm[i, j] > 0
            result += P_gldm[i, j] / (i_sq[i] * j_sq[j])
        end
    end
    return result * inv_Nz
end

"""
    gldm_small_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    Calculates the Small Dependence High Gray Level Emphasis feature.
    # Arguments
    - `P_gldm`: The GLDM matrix.
    - `ivector`: The gray level vector.
    - `jvector`: The size vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Small Dependence High Gray Level Emphasis feature value."""
function gldm_small_dependence_high_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in axes(P_gldm, 2), i in axes(P_gldm, 1)
        if P_gldm[i, j] > 0
            result += P_gldm[i, j] * i_sq[i] / j_sq[j]
        end
    end
    return result * inv_Nz
end

"""
    gldm_large_dependence_low_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    Calculates the Large Dependence Low Gray Level Emphasis feature.
    # Arguments
    - `P_gldm`: The GLDM matrix.
    - `ivector`: The gray level vector.
    - `jvector`: The size vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Large Dependence Low Gray Level Emphasis feature value.   
    """
function gldm_large_dependence_low_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in axes(P_gldm, 2), i in axes(P_gldm, 1)
        if P_gldm[i, j] > 0
            result += P_gldm[i, j] * j_sq[j] / i_sq[i]
        end
    end
    return result * inv_Nz
end

"""
    gldm_large_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    Calculates the Large Dependence High Gray Level Emphasis feature.
    # Arguments
    - `P_gldm`: The GLDM matrix.          
    - `ivector`: The gray level vector.
    - `jvector`: The size vector.
    - `Nz`: The number of zones.
    # Returns
    - The calculated Large Dependence High Gray Level Emphasis feature value."""
function gldm_large_dependence_high_gray_level_emphasis(P_gldm, i_sq, j_sq, Nz)
    inv_Nz = 1.0f0 / Nz
    result = 0.0f0
    @inbounds for j in axes(P_gldm, 2), i in axes(P_gldm, 1)
        if P_gldm[i, j] > 0
            result += P_gldm[i, j] * i_sq[i] * j_sq[j]
        end
    end
    return result * inv_Nz
end