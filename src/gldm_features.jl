
using StatsBase

"""
    get_gldm_features(img, mask, voxel_spacing; gldm_a=0, verbose=false)

Calculates and returns a dictionary of GLDM (Gray Level Dependence Matrix) features.

# Arguments
- `img`: The input image.
- `mask`: The mask defining the region of interest.
- `voxel_spacing`: The spacing of the voxels in the image.
- `gldm_a`: The alpha parameter for the GLDM calculation.
- `verbose`: If true, prints progress messages.

# Returns
- A dictionary where keys are the feature names and values are the calculated feature values.
"""
function get_gldm_features(img, mask, voxel_spacing; gldm_a=0, verbose=false)
    if verbose
        println("Calculating GLDM features...")
    end

    gldm_features = Dict{String, Float32}()

    discretized_img = discretize(img, mask)

    P_gldm, gray_levels = calculate_gldm_matrix(discretized_img, mask, gldm_a, verbose)

    Nz, pd, pg, ivector, jvector = calculate_gldm_coefficients(P_gldm, gray_levels)

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
    gldm_features["gldm_SmallDependenceLowGrayLevelEmphasis"] = gldm_small_dependence_low_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    gldm_features["gldm_SmallDependenceHighGrayLevelEmphasis"] = gldm_small_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    gldm_features["gldm_LargeDependenceLowGrayLevelEmphasis"] = gldm_large_dependence_low_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    gldm_features["gldm_LargeDependenceHighGrayLevelEmphasis"] = gldm_large_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)

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
    pd = sum(P_gldm, dims=1)
    pg = sum(P_gldm, dims=2)
    ivector = Float32.(gray_levels)
    jvector = Float32.(1:size(P_gldm, 2))

    Nz = sum(P_gldm)
    Nz = Nz == 0 ? 1.0f-6 : Nz

    return Nz, pd, pg, ivector, jvector
end

# Feature implementations
gldm_small_dependence_emphasis(pd, jvector, Nz) = sum(pd ./ (jvector' .^ 2)) / Nz
gldm_large_dependence_emphasis(pd, jvector, Nz) = sum(pd .* (jvector' .^ 2)) / Nz
gldm_gray_level_non_uniformity(pg, Nz) = sum(pg .^ 2) / Nz
gldm_dependence_non_uniformity(pd, Nz) = sum(pd .^ 2) / Nz
gldm_dependence_non_uniformity_normalized(pd, Nz) = sum(pd .^ 2) / (Nz ^ 2)
function gldm_gray_level_variance(pg, ivector, Nz)
    p_g = pg ./ Nz
    u_i = sum(p_g .* ivector)
    sum(p_g .* (ivector .- u_i) .^ 2)
end
function gldm_dependence_variance(pd, jvector, Nz)
    p_d = pd ./ Nz
    u_j = sum(p_d .* jvector')
    sum(p_d .* (jvector' .- u_j) .^ 2)
end
function gldm_dependence_entropy(P_gldm, Nz)
    p_gldm = P_gldm ./ Nz
    -sum(p_gldm .* log2.(p_gldm .+ 1.0f-16))
end
gldm_low_gray_level_emphasis(pg, ivector, Nz) = sum(pg ./ (ivector .^ 2)) / Nz
gldm_high_gray_level_emphasis(pg, ivector, Nz) = sum(pg .* (ivector .^ 2)) / Nz
function gldm_small_dependence_low_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_gldm ./ ((i_mat .^ 2) .* (j_mat .^ 2))) / Nz
end
function gldm_small_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_gldm .* (i_mat .^ 2) ./ (j_mat .^ 2)) / Nz
end
function gldm_large_dependence_low_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_gldm .* (j_mat .^ 2) ./ (i_mat .^ 2)) / Nz
end
function gldm_large_dependence_high_gray_level_emphasis(P_gldm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_gldm .* (i_mat .^ 2) .* (j_mat .^ 2)) / Nz
end
