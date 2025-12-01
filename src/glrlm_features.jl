using StatsBase

"""
        get_glrlm_features(img, mask, voxel_spacing; n_bins=nothing, bin_width=nothing, verbose=false)

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
    - `verbose`: If true, prints progress messages.

    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.

    # Examples
        # Using fixed number of bins (bin_width calculated automatically)
        features = get_glrlm_features(img, mask, spacing, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        features = get_glrlm_features(img, mask, spacing, bin_width=25.0f0)
        
        # Default (32 bins)
        features = get_glrlm_features(img, mask, spacing)
    """
function get_glrlm_features(img, mask, voxel_spacing; 
                           n_bins::Union{Int,Nothing}=nothing,
                           bin_width::Union{Float32,Nothing}=nothing,
                           verbose=false)
    if verbose
        if !isnothing(n_bins)
            println("Calcolo GLRLM con $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("Calcolo GLRLM con bin_width=$(bin_width)...")
        else
            println("Calcolo GLRLM con 32 bins (default)...")
        end
    end

    glrlm_features = Dict{String, Float32}()

    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Range intensità: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Bin width utilizzata: $(bin_width_used)")
        println("Numero di gray levels effettivi: $(n_bins_actual)")
    end

    P_glrlm, angles = calculate_glrlm_matrix(discretized_img, mask, verbose)

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
        glrlm_features["glrlm_" * feature_name] = calculate_glrlm_feature(idx, P_glrlm)
    end

    if verbose
        println("Completato! Estratte $(length(glrlm_features)) features.")
    end

    return glrlm_features
end

"""
    calculate_glrlm_matrix(discretized_img, mask, verbose)

    Calculates the Gray Level Run Length Matrix (GLRLM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLRLM matrix and the angles used for calculation.
    """
function calculate_glrlm_matrix(discretized_img, mask, verbose)
    if verbose
        println("Calculating GLRLM matrix...")
    end

    masked_img = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)
    gl_map = Dict(gl => i for (i, gl) in enumerate(gray_levels))

    max_run_length = maximum(size(discretized_img))

    angles = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, -1),
        (1, -1, -1), (-1, 1, 1)
    ]

    num_angles = length(angles)
    P_glrlm = zeros(Int, num_gl, max_run_length, num_angles)

    for (angle_idx, angle) in enumerate(angles)
        for i in eachindex(discretized_img)
            if mask[i]
                gl = discretized_img[i]
                gl_idx = gl_map[gl]

                run_length = 1
                curr_idx = i

                # Check if the run has already been counted
                prev_idx_cartesian = CartesianIndices(size(discretized_img))[curr_idx] - CartesianIndex(angle)
                if checkbounds(Bool, discretized_img, prev_idx_cartesian)
                    prev_idx = LinearIndices(size(discretized_img))[prev_idx_cartesian]
                    if mask[prev_idx] && discretized_img[prev_idx] == gl
                        continue
                    end
                end

                while true
                    next_idx_cartesian = CartesianIndices(size(discretized_img))[curr_idx] + CartesianIndex(angle)
                    if !checkbounds(Bool, discretized_img, next_idx_cartesian)
                        break
                    end
                    next_idx = LinearIndices(size(discretized_img))[next_idx_cartesian]
                    if !mask[next_idx] || discretized_img[next_idx] != gl
                        break
                    end
                    run_length += 1
                    curr_idx = next_idx
                end

                P_glrlm[gl_idx, run_length, angle_idx] += 1
            end
        end
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

    for i in 1:num_angles
        p_glrlm = P_glrlm[:, :, i]
        Nr = sum(p_glrlm)
        if Nr == 0
            continue
        end

        pr = sum(p_glrlm, dims=1)
        pg = sum(p_glrlm, dims=2)
        ivector = Float32.(1:size(p_glrlm, 1))
        jvector = Float32.(1:size(p_glrlm, 2))

        if feature_idx == 1
            feature_values[i] = sum(pr' ./ (jvector .^ 2)) / Nr
        elseif feature_idx == 2
            feature_values[i] = sum(pr' .* (jvector .^ 2)) / Nr
        elseif feature_idx == 3
            feature_values[i] = sum(pg .^ 2) / Nr
        elseif feature_idx == 4
            feature_values[i] = sum(pg .^ 2) / (Nr ^ 2)
        elseif feature_idx == 5
            feature_values[i] = sum(pr .^ 2) / Nr
        elseif feature_idx == 6
            feature_values[i] = sum(pr .^ 2) / (Nr ^ 2)
        elseif feature_idx == 7
            Np = sum(pr' .* jvector)
            feature_values[i] = Nr / Np
        elseif feature_idx == 8
            p_g = pg ./ Nr
            u_i = sum(p_g .* ivector)
            feature_values[i] = sum(p_g .* (ivector .- u_i) .^ 2)
        elseif feature_idx == 9
            p_r = pr' ./ Nr
            u_j = sum(p_r .* jvector)
            feature_values[i] = sum(p_r .* (jvector .- u_j) .^ 2)
        elseif feature_idx == 10
            p = p_glrlm ./ Nr
            feature_values[i] = -sum(p .* log2.(p .+ 1.0f-16))
        elseif feature_idx == 11
            feature_values[i] = sum(pg ./ (ivector .^ 2)) / Nr
        elseif feature_idx == 12
            feature_values[i] = sum(pg .* (ivector .^ 2)) / Nr
        elseif feature_idx == 13
            feature_values[i] = sum(p_glrlm ./ ((ivector .^ 2) .* (jvector' .^ 2))) / Nr
        elseif feature_idx == 14
            feature_values[i] = sum(p_glrlm .* (ivector .^ 2) ./ (jvector' .^ 2)) / Nr
        elseif feature_idx == 15
            feature_values[i] = sum(p_glrlm .* (jvector' .^ 2) ./ (ivector .^ 2)) / Nr
        elseif feature_idx == 16
            feature_values[i] = sum(p_glrlm .* (ivector .^ 2) .* (jvector' .^ 2)) / Nr
        end
    end

    return mean(feature_values)
end