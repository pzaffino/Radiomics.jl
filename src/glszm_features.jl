
using StatsBase

"""
    get_glszm_features(img, mask, voxel_spacing; verbose=false)

Calculates and returns a dictionary of GLSZM (Gray Level Size Zone Matrix) features.

# Arguments
- `img`: The input image.
- `mask`: The mask defining the region of interest.
- `voxel_spacing`: The spacing of the voxels in the image.
- `verbose`: If true, prints progress messages.

# Returns
- A dictionary where keys are the feature names and values are the calculated feature values.
"""
function get_glszm_features(img, mask, voxel_spacing; verbose=false)
    if verbose
        println("Calculating GLSZM features...")
    end

    glszm_features = Dict{String, Float32}()

    # 1. Discretize the image
    discretized_img = discretize(img, mask)

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


    return glszm_features
end

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
function gray_level_variance(pg, ivector, Nz)
    p_g = pg ./ Nz
    u_i = sum(p_g .* ivector)
    sum(p_g .* (ivector .- u_i) .^ 2)
end
function zone_variance(ps, jvector, Nz)
    p_s = ps' ./ Nz
    u_j = sum(p_s .* jvector)
    sum(p_s .* (jvector .- u_j) .^ 2)
end
function zone_entropy(P_glszm, Nz)
    p_glszm = P_glszm ./ Nz
    -sum(p_glszm .* log2.(p_glszm .+ 1.0f-16))
end
low_gray_level_zone_emphasis(pg, ivector, Nz) = sum(pg ./ (ivector .^ 2)) / Nz
high_gray_level_zone_emphasis(pg, ivector, Nz) = sum(pg .* (ivector .^ 2)) / Nz
function small_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm ./ ((i_mat .^ 2) .* (j_mat .^ 2))) / Nz
end
function small_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (i_mat .^ 2) ./ (j_mat .^ 2)) / Nz
end
function large_area_low_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (j_mat .^ 2) ./ (i_mat .^ 2)) / Nz
end
function large_area_high_gray_level_emphasis(P_glszm, ivector, jvector, Nz)
    i_mat = reshape(ivector, :, 1)
    j_mat = reshape(jvector, 1, :)
    sum(P_glszm .* (i_mat .^ 2) .* (j_mat .^ 2)) / Nz
end
