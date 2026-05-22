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
    - `get_raw_matrices`: If true, returns the raw GLSZM matrix.
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
function get_glszm_features(img::AbstractArray{Float64},
                             mask::BitArray,
                             voxel_spacing::Vector{Float64};
                             n_bins::Union{Int,Nothing}=nothing,
                             bin_width::Union{Float64,Nothing}=nothing,
                             get_raw_matrices::Bool=false,
                             verbose::Bool=false)::Dict{String,Any}
    if verbose
        if !isnothing(n_bins)
            println("GLSZM calculation with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("GLSZM calculation with bin_width= $(bin_width) ...")
        else
            println("GLSZM calculation with bin_width=25 (default)...")
        end
    end

    glszm_features = Dict{String, Any}()

    # 1. Discretize the image
    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    # 2. Calculate the GLSZM matrix
    P_glszm, gray_levels = calculate_glszm_matrix(discretized_img, mask, verbose)

    if get_raw_matrices
        if verbose
            println("=================================")
            println("GLSZM Matrix Dimensions: $(size(P_glszm))  →  $(size(P_glszm,1)) gray levels × $(size(P_glszm,2)) zone sizes")
            println("Number of zones: $(sum(P_glszm))")
            println("GLSZM saved in dictionary.")
            println("=================================")
        end
        glszm_features["raw_glszm_matrix"] = P_glszm
        return glszm_features
    end

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
    calculate_glszm_matrix(discretized_img::Array{Int},
                             mask::BitArray,
                             verbose::Bool)::Tuple{Matrix{Int}, Vector{Int}}

    Calculates the Gray Level Size Zone Matrix (GLSZM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLSZM matrix and the gray levels present in the ROI.
    """
function calculate_glszm_matrix(discretized_img::Array{Int},
                                 mask::BitArray,
                                 verbose::Bool)::Tuple{Matrix{Int}, Vector{Int}}

    verbose && println("Calculating GLSZM matrix...")

    sz     = size(discretized_img)
    n_dims = ndims(discretized_img)

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

    visited       = falses(sz)
    max_mask_size = count(mask)
    bfs_queue     = Vector{Int}(undef, max_mask_size)

    # Offsets CartesianIndex per i vicini (come GLCM)
    offsets = [CartesianIndex(Tuple(o)) for o in Iterators.product((-1:1 for _ in 1:n_dims)...)
               if !all(iszero, o)]

    cart_indices   = CartesianIndices(sz)
    linear_indices = LinearIndices(sz)

    zone_counts = Dict{Tuple{Int,Int}, Int}()

    @inbounds for i in eachindex(discretized_img)
        if mask[i] && !visited[i]
            gl     = discretized_img[i]
            gl_idx = gl_map[gl - min_gl + 1]

            bfs_queue[1] = i
            visited[i]   = true
            head = 1
            tail = 1

            while head <= tail
                curr_idx      = bfs_queue[head]
                curr_cart     = cart_indices[curr_idx]
                head         += 1

                for o in offsets
                    nb_cart = curr_cart + o
                    checkbounds(Bool, discretized_img, nb_cart) || continue
                    nb = linear_indices[nb_cart]
                    if mask[nb] && !visited[nb] && discretized_img[nb] == gl
                        visited[nb]  = true
                        tail        += 1
                        bfs_queue[tail] = nb
                    end
                end
            end

            zone_size = tail
            key = (gl_idx, zone_size)
            zone_counts[key] = get(zone_counts, key, 0) + 1
        end
    end

    isempty(zone_counts) && return zeros(Int, num_gl, 1), gray_levels

    max_zone_size = maximum(j for ((_, j), _) in zone_counts)
    P_glszm = zeros(Int, num_gl, max_zone_size)
    for ((gl_idx, zone_size), cnt) in zone_counts
        P_glszm[gl_idx, zone_size] += cnt
    end

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
function calculate_glszm_coefficients(P_glszm::Matrix{Int},
                                      gray_levels::Vector{Int})::Tuple{Float64,
                                                                     Float64,
                                                                     Matrix{Int},
                                                                     Matrix{Int},
                                                                     Vector{Float64},
                                                                     Vector{Float64}}
    ps = sum(P_glszm, dims=1)
    pg = sum(P_glszm, dims=2)
    ivector = Float64.(gray_levels)
    jvector = Float64.(1:size(P_glszm, 2))

    Nz = sum(P_glszm)
    Nz = Nz == 0 ? 1.0f-6 : Nz

    Np = sum(ps' .* jvector)
    Np = Np == 0 ? 1.0f-6 : Np

    return Np, Nz, ps, pg, ivector, jvector
end

# One-liners features implementations
small_area_emphasis(ps::Matrix{Int}, jvector::Vector{Float64}, Nz::Float64)::Float64 = sum(ps' ./ (jvector .^ 2)) / Nz
large_area_emphasis(ps::Matrix{Int}, jvector::Vector{Float64}, Nz::Float64)::Float64 = sum(ps' .* (jvector .^ 2)) / Nz
gray_level_non_uniformity(pg::Matrix{Int}, Nz::Float64)::Float64 = sum(pg .^ 2) / Nz
gray_level_non_uniformity_normalized(pg::Matrix{Int}, Nz::Float64)::Float64 = sum(pg .^ 2) / (Nz ^ 2)
size_zone_non_uniformity(ps::Matrix{Int}, Nz::Float64)::Float64 = sum(ps .^ 2) / Nz
size_zone_non_uniformity_normalized(ps::Matrix{Int}, Nz::Float64)::Float64 = sum(ps .^ 2) / (Nz ^ 2)
zone_percentage(Nz::Float64, Np::Float64)::Float64 = Nz / Np

"""
    gray_level_variance(pg::Matrix{Int}, ivector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Gray Level Variance feature.
    # Arguments
    - `pg`: Sum over gray levels.
    - `ivector`: Gray level vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Gray Level Variance value.
    """
function gray_level_variance(pg::Matrix{Int}, ivector::Vector{Float64}, Nz::Float64)::Float64
    inv_Nz = 1.0 / Nz
    u_i = 0.0
    @inbounds for i in eachindex(ivector)
        u_i += pg[i] * ivector[i]
    end
    u_i *= inv_Nz
    result = 0.0
    @inbounds for i in eachindex(ivector)
        diff = ivector[i] - u_i
        result += pg[i] * diff * diff
    end
    return result * inv_Nz
end

"""
    zone_variance(ps::Matrix{Int}, jvector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Zone Variance feature.
    # Arguments
    - `ps`: Sum over sizes.  
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Zone Variance value.
    """
function zone_variance(ps::Matrix{Int}, jvector::Vector{Float64}, Nz::Float64)::Float64
    inv_Nz = 1.0 / Nz
    u_j = 0.0
    @inbounds for j in eachindex(jvector)
        u_j += ps[j] * jvector[j]
    end
    u_j *= inv_Nz
    result = 0.0
    @inbounds for j in eachindex(jvector)
        diff = jvector[j] - u_j
        result += ps[j] * diff * diff
    end
    return result * inv_Nz
end

"""
    zone_entropy(P_glszm::Matrix{Int}, Nz::Float64)::Float64
    Calculates the Zone Entropy feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Zone Entropy value.
    """
function zone_entropy(P_glszm::Matrix{Int}, Nz::Float64)::Float64
    inv_Nz = 1.0 / Nz
    result = 0.0
    @inbounds for v in P_glszm
        v == 0 && continue
        p = v * inv_Nz
        result -= p * log2(p + 1.0e-16)
    end
    return result
end

low_gray_level_zone_emphasis(pg::Matrix{Int}, ivector::Vector{Float64}, Nz::Float64)::Float64 = sum(pg ./ (ivector .^ 2)) / Nz
high_gray_level_zone_emphasis(pg::Matrix{Int}, ivector::Vector{Float64}, Nz::Float64)::Float64 = sum(pg .* (ivector .^ 2)) / Nz

"""small_area_low_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Small Area Low Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Small Area Low Gray Level Emphasis value.
    """
function small_area_low_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    result = 0.0
    @inbounds for j in axes(P_glszm, 2), i in axes(P_glszm, 1)
        v = P_glszm[i, j]
        v == 0 && continue
        result += v / (ivector[i]^2 * jvector[j]^2)
    end
    return result / Nz
end

"""small_area_high_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Small Area High Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Small Area High Gray Level Emphasis value.
    """
function small_area_high_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    result = 0.0
    @inbounds for j in axes(P_glszm, 2), i in axes(P_glszm, 1)
        v = P_glszm[i, j]
        v == 0 && continue
        result += v * ivector[i]^2 / jvector[j]^2
    end
    return result / Nz
end

"""large_area_low_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Large Area Low Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Large Area Low Gray Level Emphasis value.
    """
function large_area_low_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    result = 0.0
    @inbounds for j in axes(P_glszm, 2), i in axes(P_glszm, 1)
        v = P_glszm[i, j]
        v == 0 && continue
        result += v * jvector[j]^2 / ivector[i]^2
    end
    return result / Nz
end

"""large_area_high_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    Calculates the Large Area High Gray Level Emphasis feature.
    # Arguments
    - `P_glszm`: The GLSZM matrix.
    - `ivector`: Gray level vector.
    - `jvector`: Size vector.
    - `Nz`: Number of zones.
    # Returns
    - The calculated Large Area High Gray Level Emphasis value.
    """
function large_area_high_gray_level_emphasis(P_glszm::Matrix{Int}, ivector::Vector{Float64}, jvector::Vector{Float64}, Nz::Float64)::Float64
    result = 0.0
    @inbounds for j in axes(P_glszm, 2), i in axes(P_glszm, 1)
        v = P_glszm[i, j]
        v == 0 && continue
        result += v * ivector[i]^2 * jvector[j]^2
    end
    return result / Nz
end
