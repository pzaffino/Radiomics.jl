using StatsBase

"""
    get_glrlm_features with weighting support

    Calculates and returns a dictionary of GLRLM (Gray Level Run Length Matrix) features.

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
    - `weighting_norm`: Weighting method ("infinity", "euclidean", "manhattan", "no_weighting", or nothing for no weighting)
    - `verbose`: If true, prints progress messages.

    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.

    # Examples
        # Using fixed number of bins with weighting
        features = get_glrlm_features(img, mask, spacing, n_bins=64, weighting_norm="euclidean")
        
        # Using fixed bin width with weighting
        features = get_glrlm_features(img, mask, spacing, bin_width=25.0f0, weighting_norm="infinity")
        
        # Default (32 bins, no weighting)
        features = get_glrlm_features(img, mask, spacing)
"""
function get_glrlm_features(img, mask, voxel_spacing;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    verbose=false)
    
    if verbose
        if !isnothing(n_bins)
            println("Calculating GLRLM with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("Calculating GLRLM with bin_width=$(bin_width)...")
        else
            println("GLRLM calculation with 32 bins (default)...")
        end
        if !isnothing(weighting_norm)
            println("Applying weighting: $(weighting_norm)")
        end
    end

    glrlm_features = Dict{String,Float32}()

    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective Gray level utilized: $(n_bins_actual)")
    end

    P_glrlm, angles = calculate_glrlm_matrix(discretized_img, mask, voxel_spacing, weighting_norm, verbose)

    feature_names = [
        "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
        "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance",
        "RunVariance", "RunEntropy", "LowGrayLevelRunEmphasis",
        "HighGrayLevelRunEmphasis", "ShortRunLowGrayLevelEmphasis",
        "ShortRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
        "LongRunHighGrayLevelEmphasis"
    ]

    for (idx, feature_name) in enumerate(feature_names)
        glrlm_features["glrlm_"*feature_name] = calculate_glrlm_feature(idx, P_glrlm)
    end

    if verbose
        println("Completed! Extract $(length(glrlm_features)) features.")
    end

    return glrlm_features
end

"""
    calculate_glrlm_matrix with weighting support

    Calculates the Gray Level Run Length Matrix (GLRLM) with optional weighting.

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `voxel_spacing`: The spacing of the voxels in the image.
    - `weighting_norm`: Weighting method (nothing, "infinity", "euclidean", "manhattan", "no_weighting").
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLRLM matrix and the angles used for calculation.
"""
function calculate_glrlm_matrix(discretized_img, mask, voxel_spacing, weighting_norm, verbose)
    if verbose
        println("Calculating GLRLM matrix...")
    end

    masked_img = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)

    # Pre-allocate gray level map with sizehint!
    gl_map = Dict{Int,Int}()
    sizehint!(gl_map, num_gl)
    @inbounds for (i, gl) in enumerate(gray_levels)
        gl_map[gl] = i
    end

    max_run_length = maximum(size(discretized_img))

    angles = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, -1),
        (1, -1, -1), (-1, 1, 1)
    ]

    num_angles = length(angles)
    P_glrlm = zeros(Float32, num_gl, max_run_length, num_angles)

    # Pre-compute CartesianIndices and LinearIndices
    cart_indices = CartesianIndices(size(discretized_img))
    lin_indices = LinearIndices(size(discretized_img))

    @inbounds for (angle_idx, angle) in enumerate(angles)
        for i in eachindex(discretized_img)
            if mask[i]
                gl = discretized_img[i]
                gl_idx = gl_map[gl]

                run_length = 1
                curr_idx = i

                # Check if the run has already been counted
                prev_idx_cartesian = cart_indices[curr_idx] - CartesianIndex(angle)
                if checkbounds(Bool, discretized_img, prev_idx_cartesian)
                    prev_idx = lin_indices[prev_idx_cartesian]
                    if mask[prev_idx] && discretized_img[prev_idx] == gl
                        continue
                    end
                end

                while true
                    next_idx_cartesian = cart_indices[curr_idx] + CartesianIndex(angle)
                    if !checkbounds(Bool, discretized_img, next_idx_cartesian)
                        break
                    end
                    next_idx = lin_indices[next_idx_cartesian]
                    if !mask[next_idx] || discretized_img[next_idx] != gl
                        break
                    end
                    run_length += 1
                    curr_idx = next_idx
                end

                P_glrlm[gl_idx, run_length, angle_idx] += 1.0f0
            end
        end
    end

    # Apply weighting if specified
    if !isnothing(weighting_norm)
        if verbose
            println("Applying weighting ($(weighting_norm))...")
        end
        
        pixel_spacing = voxel_spacing
        weights = ones(Float32, num_angles)
        
        @inbounds for (a_idx, angle) in enumerate(angles)
            angle_vec = Float32.([angle[1], angle[2], angle[3]])
            abs_angle = abs.(angle_vec)
            
            if weighting_norm == "infinity"
                weights[a_idx] = maximum(abs_angle .* pixel_spacing)
            elseif weighting_norm == "euclidean"
                weights[a_idx] = sqrt(sum((abs_angle .* pixel_spacing).^2))
            elseif weighting_norm == "manhattan"
                weights[a_idx] = sum(abs_angle .* pixel_spacing)
            elseif weighting_norm == "no_weighting"
                weights[a_idx] = 1.0f0
            else
                @warn "Weighting norm \"$(weighting_norm)\" is unknown, weighting factor set to 1"
                weights[a_idx] = 1.0f0
            end
        end
        
        if verbose
            println("Weights computed: ", weights)
        end
        
        # Apply weights and sum across angles
        # P_glrlm_weighted = sum(P_glrlm .* reshape(weights, 1, 1, :), dims=3)
        weighted_glrlm = zeros(Float32, num_gl, max_run_length, 1)
        @inbounds for angle_idx in 1:num_angles
            for j in 1:max_run_length
                for i in 1:num_gl
                    weighted_glrlm[i, j, 1] += P_glrlm[i, j, angle_idx] * weights[angle_idx]
                end
            end
        end
        
        P_glrlm = weighted_glrlm
    end

    return P_glrlm, angles
end

"""
    calculate_glrlm_feature(feature_idx, P_glrlm)

    Calculates a specific GLRLM feature.

    # Arguments
    - `feature_idx`: The index of the feature to calculate.
    - `P_glrlm`: The GLRLM matrix.

    # Returns
    - The calculated feature value.
"""
function calculate_glrlm_feature(feature_idx, P_glrlm)
    num_angles = size(P_glrlm, 3)
    feature_values = zeros(Float32, num_angles)

    @inbounds for i in 1:num_angles
        p_glrlm = P_glrlm[:, :, i]
        Nr = sum(p_glrlm)
        if Nr == 0
            continue
        end

        # Pre-compute inverse of Nr
        inv_Nr = 1.0f0 / Float32(Nr)
        inv_Nr_sq = inv_Nr * inv_Nr

        pr = sum(p_glrlm, dims=1)
        pg = sum(p_glrlm, dims=2)
        ivector = Float32.(1:size(p_glrlm, 1))
        jvector = Float32.(1:size(p_glrlm, 2))

        # Pre-compute squared vectors
        ivector_sq = ivector .^ 2
        jvector_sq = jvector .^ 2

        if feature_idx == 1
            # ShortRunEmphasis
            feature_values[i] = sum(pr' ./ jvector_sq) * inv_Nr
        elseif feature_idx == 2
            # LongRunEmphasis
            feature_values[i] = sum(pr' .* jvector_sq) * inv_Nr
        elseif feature_idx == 3
            # GrayLevelNonUniformity
            feature_values[i] = sum(pg .^ 2) * inv_Nr
        elseif feature_idx == 4
            # GrayLevelNonUniformityNormalized
            feature_values[i] = sum(pg .^ 2) * inv_Nr_sq
        elseif feature_idx == 5
            # RunLengthNonUniformity
            feature_values[i] = sum(pr .^ 2) * inv_Nr
        elseif feature_idx == 6
            # RunLengthNonUniformityNormalized
            feature_values[i] = sum(pr .^ 2) * inv_Nr_sq
        elseif feature_idx == 7
            # RunPercentage
            Np = sum(pr' .* jvector)
            feature_values[i] = Float32(Nr) / Np
        elseif feature_idx == 8
            # GrayLevelVariance
            p_g = pg .* inv_Nr
            u_i = sum(p_g .* ivector)
            feature_values[i] = sum(p_g .* (ivector .- u_i) .^ 2)
        elseif feature_idx == 9
            # RunVariance
            p_r = pr' .* inv_Nr
            u_j = sum(p_r .* jvector)
            feature_values[i] = sum(p_r .* (jvector .- u_j) .^ 2)
        elseif feature_idx == 10
            # RunEntropy
            p = p_glrlm .* inv_Nr
            feature_values[i] = -sum(p .* log2.(p .+ 1.0f-16))
        elseif feature_idx == 11
            # LowGrayLevelRunEmphasis
            feature_values[i] = sum(pg ./ ivector_sq) * inv_Nr
        elseif feature_idx == 12
            # HighGrayLevelRunEmphasis
            feature_values[i] = sum(pg .* ivector_sq) * inv_Nr
        elseif feature_idx == 13
            # ShortRunLowGrayLevelEmphasis
            feature_values[i] = sum(p_glrlm ./ (ivector_sq .* jvector_sq')) * inv_Nr
        elseif feature_idx == 14
            # ShortRunHighGrayLevelEmphasis
            feature_values[i] = sum(p_glrlm .* ivector_sq ./ jvector_sq') * inv_Nr
        elseif feature_idx == 15
            # LongRunLowGrayLevelEmphasis
            feature_values[i] = sum(p_glrlm .* jvector_sq' ./ ivector_sq) * inv_Nr
        elseif feature_idx == 16
            # LongRunHighGrayLevelEmphasis
            feature_values[i] = sum(p_glrlm .* ivector_sq .* jvector_sq') * inv_Nr
        end
    end

    return mean(feature_values)
end