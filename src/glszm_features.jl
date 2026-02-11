using StatsBase

"""
    get_glszm_features(img, mask, voxel_spacing; n_bins=nothing, bin_width=nothing, verbose=false)

    Calculates and returns a dictionary of GLSZM (Gray Level Size Zone Matrix) features.

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
        features = get_glszm_features(img, mask, spacing, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        features = get_glszm_features(img, mask, spacing, bin_width=25.0f0)
        
        # Default (32 bins)
        features = get_glszm_features(img, mask, spacing)
    """
function get_glszm_features(img, mask, voxel_spacing; 
                           n_bins::Union{Int,Nothing}=nothing,
                           bin_width::Union{Float32,Nothing}=nothing,
                           verbose=false)
    if verbose
        if !isnothing(n_bins)
            println("GLSZM calculation with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("GLSZM calculation with bin_width= $(bin_width) ...")
        else
            println("GLSZM calculation with bin_width=25 (default)...")
        end
    end

    glszm_features = Dict{String, Float32}()

    # 1. Discretize the image
    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective Gray level utilized: $(n_bins_actual)")
    end

    # 2. Calculate the GLSZM matrix
    P_glszm, gray_levels = calculate_glszm_matrix(discretized_img, mask, verbose)

    # 3. Calculate coefficients
    Np, Nz, ps, pg, ivector, jvector = calculate_glszm_coefficients(P_glszm, gray_levels)

    # 4. Calculate features
    glszm_features["glszm_SmallAreaEmphasis"] = small_area_emphasis(ps, jvector, Nz)
    glszm_features["glszm_LargeAreaEmphasis"] = large_area_emphasis(ps, jvector, Nz)
    glszm_features["glszm_GrayLevelNonUniformity"] = gray_level_non_uniformity(pg, Nz)
    glszm_features["glszm_GrayLevelNonUniformityNormalized"] = gray_level_non_uniformity_normalized(pg, Nz)
    glszm_features["glszm_SizeZoneNonUniformity"] = size_zone_non_uniformity(ps, Nz)
    glszm_features["glszm_SizeZoneNonUniformityNormalized"] = size_zone_non_uniformity_normalized(ps, Nz)
    glszm_features["glszm_ZonePercentage"] = zone_percentage(Nz, Np)
    glszm_features["glszm_GrayLevelVariance"] = gray_level_variance(pg, ivector, Nz)
    glszm_features["glszm_ZoneVariance"] = zone_variance(ps, jvector, Nz)
    glszm_features["glszm_ZoneEntropy"] = zone_entropy(P_glszm, Nz)
    glszm_features["glszm_LowGrayLevelZoneEmphasis"] = low_gray_level_zone_emphasis(pg, ivector, Nz)
    glszm_features["glszm_HighGrayLevelZoneEmphasis"] = high_gray_level_zone_emphasis(pg, ivector, Nz)
    glszm_features["glszm_SmallAreaLowGrayLevelEmphasis"] = small_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    glszm_features["glszm_SmallAreaHighGrayLevelEmphasis"] = small_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    glszm_features["glszm_LargeAreaLowGrayLevelEmphasis"] = large_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    glszm_features["glszm_LargeAreaHighGrayLevelEmphasis"] = large_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)

    if verbose
        println("Completed! $(length(glszm_features)) features esxtacted.")
    end

    return glszm_features
end

"""
    calculate_glszm_matrix(discretized_img, mask, verbose)

    Calculates the Gray Level Size Zone Matrix (GLSZM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLSZM matrix and the gray levels present in the ROI.
    """
function calculate_glszm_matrix(discretized_img, mask, verbose)
    if verbose
        println("Calculating GLSZM matrix...")
    end

    masked_img = discretized_img[mask]
    min_gl = minimum(masked_img)
    max_gl = maximum(masked_img)
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)
    gl_map = Dict(gl => i for (i, gl) in enumerate(gray_levels))

    max_size = count(mask)
    P_glszm = zeros(Int, num_gl, max_size)

    visited = falses(size(discretized_img))

    for i in eachindex(discretized_img)
        if mask[i] && !visited[i]
            gl = discretized_img[i]
            gl_idx = gl_map[gl]

            zone_size = 0
            q = [i]
            visited[i] = true

            head = 1
            while head <= length(q)
                curr_idx = q[head]
                head += 1
                zone_size += 1

                for neighbor_idx in get_neighbors(curr_idx, size(discretized_img))
                    if mask[neighbor_idx] && !visited[neighbor_idx] && discretized_img[neighbor_idx] == gl
                        visited[neighbor_idx] = true
                        push!(q, neighbor_idx)
                    end
                end
            end

            if zone_size > 0
                P_glszm[gl_idx, zone_size] += 1
            end
        end
    end

    # Trim empty columns
    last_col = findlast(sum(P_glszm, dims=1) .> 0)
    last_col = last_col === nothing ? 0 : last_col[2]
    P_glszm = P_glszm[:, 1:last_col]

    return P_glszm, gray_levels
end

"""
    calculate_glszm_coefficients(P_glszm, gray_levels)

    Calculates the coefficients used in the GLSZM feature calculations.

    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `gray_levels`: The gray levels present in the ROI.

    # Returns
    - A tuple containing the number of voxels, number of zones, sum over gray levels, sum over sizes, gray level vector, and size vector.
    """
function calculate_glszm_coefficients(P_glszm, gray_levels)
    ps = sum(P_glszm, dims=1)
    pg = sum(P_glszm, dims=2)
    ivector = Float32.(gray_levels)
    jvector = Float32.(1:size(P_glszm, 2))

    Nz = sum(P_glszm)
    Nz = Nz == 0 ? 1.0f-6 : Nz

    Np = sum(ps' .* jvector)
    Np = Np == 0 ? 1.0f-6 : Np

    return Np, Nz, ps, pg, ivector, jvector
end

# Feature implementations
small_area_emphasis(ps, jvector, Nz) = sum(ps' ./ (jvector .^ 2)) / Nz
large_area_emphasis(ps, jvector, Nz) = sum(ps' .* (jvector .^ 2)) / Nz
gray_level_non_uniformity(pg, Nz) = sum(pg .^ 2) / Nz
gray_level_non_uniformity_normalized(pg, Nz) = sum(pg .^ 2) / (Nz ^ 2)
size_zone_non_uniformity(ps, Nz) = sum(ps .^ 2) / Nz
size_zone_non_uniformity_normalized(ps, Nz) = sum(ps .^ 2) / (Nz ^ 2)
zone_percentage(Nz, Np) = Nz / Np

"""
    gray_level_variance(pg, ivector, Nz)
    Calculates the Gray Level Variance feature.
    # Arguments
    - `pg`: Sum over gray levels.
    - `ivector`: Gray level vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Gray Level Variance value.
    """
function gray_level_variance(pg, ivector, Nz)
    p_g = pg ./ Nz
    u_i = sum(p_g .* ivector)
    sum(p_g .* (ivector .- u_i) .^ 2)
end

"""
    zone_variance(ps, jvector, Nz)
    Calculates the Zone Variance feature.
    # Arguments
    - `ps`: Sum over sizes.  
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Zone Variance value.
    """
function zone_variance(ps, jvector, Nz)
    p_s = ps' ./ Nz
    u_j = sum(p_s .* jvector)
    sum(p_s .* (jvector .- u_j) .^ 2)
end

"""
    zone_entropy(P_glszm, Nz)
    Calculates the Zone Entropy feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Zone Entropy value.
    """
function zone_entropy(P_glszm, Nz)
    p_glszm = P_glszm ./ Nz
    -sum(p_glszm .* log2.(p_glszm .+ 1.0f-16))
end

low_gray_level_zone_emphasis(pg, ivector, Nz) = sum(pg ./ (ivector .^ 2)) / Nz
high_gray_level_zone_emphasis(pg, ivector, Nz) = sum(pg .* (ivector .^ 2)) / Nz

"""small_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    Calculates the Small Area Low Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Small Area Low Gray Level Emphasis value.
    """
function small_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm ./ ((i_mat .^ 2) .* (j_mat .^ 2))) / Nz
end

"""small_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    Calculates the Small Area High Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Small Area High Gray Level Emphasis value.
    """
function small_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (i_mat .^ 2) ./ (j_mat .^ 2)) / Nz
end

"""large_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    Calculates the Large Area Low Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Large Area Low Gray Level Emphasis value.
    """
function large_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (j_mat .^ 2) ./ (i_mat .^ 2)) / Nz
end

"""large_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    Calculates the Large Area High Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Large Area High Gray Level Emphasis value.
    """
function large_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (i_mat .^ 2) .* (j_mat .^ 2)) / Nz
end
