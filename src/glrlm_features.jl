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
    - `get_raw_matrices`: If true, returns the raw GLRLM matrix.
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
    bin_width::Union{Float64,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    get_raw_matrices::Bool=false,
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

    glrlm_features = Dict{String,Any}()

    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective Gray level utilized: $(n_bins_actual)")
    end

    P_glrlm, angles = calculate_glrlm_matrix(discretized_img, mask, voxel_spacing, weighting_norm, verbose)

    if get_raw_matrices
        if verbose
            println("=================================")
            println("GLRLM Matrix Dimensions: $(size(P_glrlm))  →  $(size(P_glrlm,1)) gray levels × $(size(P_glrlm,2)) run lengths")
            println("Number of runs: $(sum(P_glrlm))")
            println("GLRLM saved in dictionary.")
            println("=================================")
        end
        glrlm_features["raw_glrlm_matrix"] = P_glrlm, angles
        return glrlm_features
    end

    feature_names = [
        "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
        "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance",
        "RunVariance", "RunEntropy", "LowGrayLevelRunEmphasis",
        "HighGrayLevelRunEmphasis", "ShortRunLowGrayLevelEmphasis",
        "ShortRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
        "LongRunHighGrayLevelEmphasis"
    ]

    extracted_features = extract_all_glrlm_features(P_glrlm, gray_levels, feature_names)
    merge!(glrlm_features, extracted_features)

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

    dim = ndims(discretized_img)

    masked_img = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl = length(gray_levels)

    gl_map = Dict{Int,Int}()
    sizehint!(gl_map, num_gl)
    @inbounds for (i, gl) in enumerate(gray_levels)
        gl_map[gl] = i
    end

    max_run_length = maximum(size(discretized_img))

    if dim == 2
        angles = [
            (1, 0), (0, 1), (1, 1), (1, -1),
            (-1, 0), (0, -1), (-1, -1), (-1, 1)
        ]
    else
        angles = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1),
            (1, 0, -1), (-1, 0, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
            (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, -1),
            (1, -1, -1), (-1, 1, 1)
        ]
    end

    num_angles = length(angles)
    P_glrlm = zeros(Float32, num_gl, max_run_length, num_angles)

    cart_indices = CartesianIndices(size(discretized_img))
    lin_indices = LinearIndices(size(discretized_img))

    # Pre-compute CartesianIndex for each angle
    c_angles = [CartesianIndex(a) for a in angles]

    # Iterate only over masked voxels instead of the entire image
    masked_cart_indices = cart_indices[mask]

    @inbounds for (angle_idx, c_angle) in enumerate(c_angles)
        for curr_idx_cart in masked_cart_indices
            i = lin_indices[curr_idx_cart]
            gl = discretized_img[i]
            gl_idx = gl_map[gl]

            # Skip if this voxel is not the start of a run
            prev_idx_cart = curr_idx_cart - c_angle
            if checkbounds(Bool, discretized_img, prev_idx_cart)
                prev_idx = lin_indices[prev_idx_cart]
                if mask[prev_idx] && discretized_img[prev_idx] == gl
                    continue
                end
            end

            # Count run length
            run_length = 1
            next_idx_cart = curr_idx_cart + c_angle
            while checkbounds(Bool, discretized_img, next_idx_cart) &&
                  mask[lin_indices[next_idx_cart]] &&
                  discretized_img[next_idx_cart] == gl
                run_length += 1
                next_idx_cart += c_angle
            end

            P_glrlm[gl_idx, run_length, angle_idx] += 1.0f0
        end
    end

    # Weighted sum in-place without allocating intermediate arrays
    if !isnothing(weighting_norm)
        weights = ones(Float32, num_angles)
        @inbounds for (a_idx, angle) in enumerate(angles)
            angle_vec = Float32.([angle...])
            current_spacing = voxel_spacing[1:dim]
            abs_angle_dist = abs.(angle_vec) .* current_spacing
            if weighting_norm == "infinity"
                weights[a_idx] = maximum(abs_angle_dist)
            elseif weighting_norm == "euclidean"
                weights[a_idx] = sqrt(sum(abs_angle_dist .^ 2))
            elseif weighting_norm == "manhattan"
                weights[a_idx] = sum(abs_angle_dist)
            end
        end

        P_out = zeros(Float32, num_gl, max_run_length)
        @inbounds for a_idx in 1:num_angles
            @views P_out .+= weights[a_idx] .* P_glrlm[:, :, a_idx]
        end
        return reshape(P_out, num_gl, max_run_length, 1), angles
    end

    return P_glrlm, angles
end

"""
    extract_all_glrlm_features(P_glrlm, gray_levels, feature_names)

    Calculates all GLRLM features in a single pass to avoid redundant computations.

    # Arguments
    - `P_glrlm`: The GLRLM matrix.
    - `gray_levels`: The gray levels used.
    - `feature_names`: The list of feature names to extract.

    # Returns
    - A dictionary containing all calculated feature values.
"""
function extract_all_glrlm_features(P_glrlm, gray_levels, feature_names)
    num_gl = length(gray_levels)
    num_angles = size(P_glrlm, 3)
    max_run_length = size(P_glrlm, 2)
    num_features = length(feature_names)
    feature_sums = zeros(Float32, num_features)

    ivector = Float32.(gray_levels)
    jvector = Float32.(1:max_run_length)
    # Pre-compute squared vectors outside the loop (constant across angles)
    ivector_sq = ivector .^ 2
    jvector_sq = jvector .^ 2

    # Pre-allocate marginal sum buffers outside the loop to avoid repeated allocations
    pr = zeros(Float32, 1, max_run_length)
    pg = zeros(Float32, num_gl, 1)

    @inbounds for i in 1:num_angles
        p_slice = @view P_glrlm[:, :, i]
        Nr = sum(p_slice)
        if Nr == 0
            continue
        end

        inv_Nr = 1.0f0 / Float32(Nr)
        inv_Nr_sq = inv_Nr * inv_Nr

        # In-place marginal sums: zero allocations
        sum!(pr, p_slice)
        sum!(pg, p_slice)

        # 1. ShortRunEmphasis
        feature_sums[1] += sum(pr[j] / jvector_sq[j] for j in 1:max_run_length) * inv_Nr
        # 2. LongRunEmphasis
        feature_sums[2] += sum(pr[j] * jvector_sq[j] for j in 1:max_run_length) * inv_Nr
        # 3. GrayLevelNonUniformity
        feature_sums[3] += sum(pg[g]^2 for g in 1:num_gl) * inv_Nr
        # 4. GrayLevelNonUniformityNormalized
        feature_sums[4] += sum(pg[g]^2 for g in 1:num_gl) * inv_Nr_sq
        # 5. RunLengthNonUniformity
        feature_sums[5] += sum(pr[j]^2 for j in 1:max_run_length) * inv_Nr
        # 6. RunLengthNonUniformityNormalized
        feature_sums[6] += sum(pr[j]^2 for j in 1:max_run_length) * inv_Nr_sq
        # 7. RunPercentage
        Np = sum(pr[j] * jvector[j] for j in 1:max_run_length)
        feature_sums[7] += Float32(Nr) / Np
        # 8. GrayLevelVariance
        u_i = sum(pg[g] * ivector[g] for g in 1:num_gl) * inv_Nr
        feature_sums[8] += sum(pg[g] * inv_Nr * (ivector[g] - u_i)^2 for g in 1:num_gl)
        # 9. RunVariance
        u_j = sum(pr[j] * jvector[j] for j in 1:max_run_length) * inv_Nr
        feature_sums[9] += sum(pr[j] * inv_Nr * (jvector[j] - u_j)^2 for j in 1:max_run_length)
        # 10. RunEntropy
        feature_sums[10] += -sum(p_slice[g,j] * inv_Nr * log2(p_slice[g,j] * inv_Nr + 1.0f-16)
                                  for g in 1:num_gl, j in 1:max_run_length)
        # 11. LowGrayLevelRunEmphasis
        feature_sums[11] += sum(pg[g] / ivector_sq[g] for g in 1:num_gl) * inv_Nr
        # 12. HighGrayLevelRunEmphasis
        feature_sums[12] += sum(pg[g] * ivector_sq[g] for g in 1:num_gl) * inv_Nr
        # 13. ShortRunLowGrayLevelEmphasis
        feature_sums[13] += sum(p_slice[g,j] / (ivector_sq[g] * jvector_sq[j])
                                 for g in 1:num_gl, j in 1:max_run_length) * inv_Nr
        # 14. ShortRunHighGrayLevelEmphasis
        feature_sums[14] += sum(p_slice[g,j] * ivector_sq[g] / jvector_sq[j]
                                 for g in 1:num_gl, j in 1:max_run_length) * inv_Nr
        # 15. LongRunLowGrayLevelEmphasis
        feature_sums[15] += sum(p_slice[g,j] * jvector_sq[j] / ivector_sq[g]
                                 for g in 1:num_gl, j in 1:max_run_length) * inv_Nr
        # 16. LongRunHighGrayLevelEmphasis
        feature_sums[16] += sum(p_slice[g,j] * ivector_sq[g] * jvector_sq[j]
                                 for g in 1:num_gl, j in 1:max_run_length) * inv_Nr
    end

    inv_angles = 1.0f0 / num_angles
    glrlm_features = Dict{String,Any}()
    for (idx, name) in enumerate(feature_names)
        glrlm_features["glrlm_"*name] = feature_sums[idx] * inv_angles
    end

    return glrlm_features
end