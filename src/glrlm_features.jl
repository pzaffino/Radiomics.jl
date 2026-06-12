using StatsBase

"""
    get_glrlm_features with weighting support

    Calculates and returns a dictionary of GLRLM (Gray Level Run Length Matrix) features.
"""
function get_glrlm_features(img::AbstractArray{Float64},
                             mask::BitArray,
                             voxel_spacing::Vector{Float64};
                             n_bins::Union{Int,Nothing}=nothing,
                             bin_width::Union{Float64,Nothing}=nothing,
                             weighting_norm::Union{String,Nothing}=nothing,
                             get_raw_matrices::Bool=false,
                             features_std::Bool=false,
                             verbose::Bool=false)::Dict{String,Any}
    
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
        if features_std
            println("Computing std across angles: enabled")
        end
    end

    glrlm_features = Dict{String,Any}()

    # Assuming discretize_image is defined in utils.jl or globally available within the module
    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

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

    extracted_features = extract_all_glrlm_features(P_glrlm, gray_levels, feature_names; features_std=features_std)
    merge!(glrlm_features, extracted_features)

    if verbose
        println("Completed! Extracted $(length(glrlm_features)) features.")
    end

    return glrlm_features
end

const Angle = Union{Tuple{Int,Int}, Tuple{Int,Int,Int}}

"""
    calculate_glrlm_matrix

    Calculates the GLRLM matrix. Tracks the real maximum run length 
    dynamically to eliminate downstream computations on empty trailing columns.
"""
function calculate_glrlm_matrix(discretized_img::Array{Int},
                                 mask::BitArray,
                                 voxel_spacing::Vector{Float64},
                                 weighting_norm::Union{String,Nothing},
                                 verbose::Bool)::Tuple{Array{Float64,3}, Vector{Angle}}
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

    max_run_length_possible = maximum(size(discretized_img))

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
    P_glrlm = zeros(Float64, num_gl, max_run_length_possible, num_angles)

    cart_indices = CartesianIndices(size(discretized_img))
    c_angles = [CartesianIndex(a) for a in angles]
    masked_cart_indices = cart_indices[mask]

    # Track the actual maximum run length to scale down the feature extraction phase
    actual_max_run = 1

    @inbounds for (angle_idx, c_angle) in enumerate(c_angles)
        for curr_idx_cart in masked_cart_indices
            gl = discretized_img[curr_idx_cart]
            gl_idx = gl_map[gl]

            # Check if this voxel is the start of a run
            prev_idx_cart = curr_idx_cart - c_angle
            if checkbounds(Bool, discretized_img, prev_idx_cart)
                # Direct indexing using CartesianIndex bypasses LinearIndices overhead
                if mask[prev_idx_cart] && discretized_img[prev_idx_cart] == gl
                    continue
                end
            end

            # Count run length along current direction vector
            run_length = 1
            next_idx_cart = curr_idx_cart + c_angle
            while checkbounds(Bool, discretized_img, next_idx_cart) &&
                  mask[next_idx_cart] &&
                  discretized_img[next_idx_cart] == gl
                run_length += 1
                next_idx_cart += c_angle
            end

            P_glrlm[gl_idx, run_length, angle_idx] += 1.0
            if run_length > actual_max_run
                actual_max_run = run_length
            end
        end
    end

    # Crop the matrix to the actual maximum run length found
    P_glrlm = P_glrlm[:, 1:actual_max_run, :]

    if !isnothing(weighting_norm)
        weights = ones(Float64, num_angles)
        @inbounds for (a_idx, angle) in enumerate(angles)
            # Tuple unpacking prevents unnecessary heap allocations
            if dim == 2
                dists = (abs(angle[1]) * voxel_spacing[1], abs(angle[2]) * voxel_spacing[2])
            else
                dists = (abs(angle[1]) * voxel_spacing[1], abs(angle[2]) * voxel_spacing[2], abs(angle[3]) * voxel_spacing[3])
            end
            
            if weighting_norm == "infinity"
                weights[a_idx] = max(dists...)
            elseif weighting_norm == "euclidean"
                weights[a_idx] = sqrt(sum(x -> x^2, dists))
            elseif weighting_norm == "manhattan"
                weights[a_idx] = sum(dists)
            end
        end

        P_out = zeros(Float64, num_gl, actual_max_run)
        @inbounds for a_idx in 1:num_angles
            @views P_out .+= weights[a_idx] .* P_glrlm[:, :, a_idx]
        end
        return reshape(P_out, num_gl, actual_max_run, 1), angles
    end

    return P_glrlm, angles
end

"""
    extract_all_glrlm_features

    Extracts all features via a high-performance single-pass, flat 2D loop.
    Skips empty rows/columns and eliminates nested generator heap allocations.

    If `features_std=true`, also computes the standard deviation of each feature
    across angles, stored as "glrlm_<FeatureName>_std".
    Note: when weighting_norm is used, num_angles == 1 so std will be 0 for all features.
"""
function extract_all_glrlm_features(P_glrlm::Array{Float64,3},
                                     gray_levels::Vector{Int},
                                     feature_names::Vector{String};
                                     features_std::Bool=false)::Dict{String,Any}
    num_gl         = length(gray_levels)
    max_run_length = size(P_glrlm, 2)
    num_angles     = size(P_glrlm, 3)
    num_features   = length(feature_names)

    feature_sums    = zeros(Float64, num_features)
    feature_sums_sq = zeros(Float64, num_features)  # used only when features_std=true
    feature_min     = fill(Inf, num_features)   
    feature_max     = fill(-Inf, num_features)

    ivector    = Float64.(gray_levels)
    jvector    = Float64.(1:max_run_length)
    ivector_sq = ivector .^ 2
    jvector_sq = jvector .^ 2

    pr = zeros(Float64, 1, max_run_length)
    pg = zeros(Float64, num_gl, 1)

    # Temporary buffer for per-angle feature values (allocated once, reused)
    angle_vals = zeros(Float64, num_features)

    @inbounds for a in 1:num_angles
        p_slice = @view P_glrlm[:, :, a]
        Nr = sum(p_slice)
        if Nr == 0.0
            continue
        end

        fill!(angle_vals, 0.0)  # reset per-angle buffer

        inv_Nr    = 1.0 / Nr
        inv_Nr_sq = inv_Nr * inv_Nr

        sum!(pr, p_slice)
        sum!(pg, p_slice)

        # 1D: Run Length metrics (j-dependent)
        Np      = 0.0
        u_j_sum = 0.0
        for j in 1:max_run_length
            if pr[j] > 0.0
                j_sq = jvector_sq[j]
                pr_j = pr[j]

                angle_vals[1] += (pr_j / j_sq) * inv_Nr          # ShortRunEmphasis
                angle_vals[2] += (pr_j * j_sq) * inv_Nr           # LongRunEmphasis
                angle_vals[5] += (pr_j^2) * inv_Nr                # RunLengthNonUniformity
                angle_vals[6] += (pr_j^2) * inv_Nr_sq             # RunLengthNonUniformityNormalized

                Np      += pr_j * jvector[j]
                u_j_sum += pr_j * jvector[j]
            end
        end

        if Np > 0.0
            angle_vals[7] += Nr / Np                              # RunPercentage
        end

        # 1D: Gray Level metrics (g-dependent)
        u_i_sum = 0.0
        for g in 1:num_gl
            if pg[g] > 0.0
                i_sq  = ivector_sq[g]
                pg_g  = pg[g]

                angle_vals[3]  += (pg_g^2) * inv_Nr               # GrayLevelNonUniformity
                angle_vals[4]  += (pg_g^2) * inv_Nr_sq            # GrayLevelNonUniformityNormalized
                angle_vals[11] += (pg_g / i_sq) * inv_Nr          # LowGrayLevelRunEmphasis
                angle_vals[12] += (pg_g * i_sq) * inv_Nr          # HighGrayLevelRunEmphasis

                u_i_sum += pg_g * ivector[g]
            end
        end

        # Variance profiles
        u_j = u_j_sum * inv_Nr
        u_i = u_i_sum * inv_Nr
        for j in 1:max_run_length
            if pr[j] > 0.0
                angle_vals[9] += pr[j] * inv_Nr * (jvector[j] - u_j)^2  # RunVariance
            end
        end
        for g in 1:num_gl
            if pg[g] > 0.0
                angle_vals[8] += pg[g] * inv_Nr * (ivector[g] - u_i)^2  # GrayLevelVariance
            end
        end

        # Unified 2D loop (column-major for cache locality)
        for j in 1:max_run_length
            j_sq = jvector_sq[j]
            for g in 1:num_gl
                p_val = p_slice[g, j]
                if p_val > 0.0
                    i_sq = ivector_sq[g]
                    prob = p_val * inv_Nr

                    angle_vals[10] += -prob * log2(prob + 1e-16)              # RunEntropy
                    angle_vals[13] += (p_val / (i_sq * j_sq)) * inv_Nr        # ShortRunLowGrayLevelEmphasis
                    angle_vals[14] += (p_val * i_sq / j_sq) * inv_Nr          # ShortRunHighGrayLevelEmphasis
                    angle_vals[15] += (p_val * j_sq / i_sq) * inv_Nr          # LongRunLowGrayLevelEmphasis
                    angle_vals[16] += (p_val * i_sq * j_sq) * inv_Nr          # LongRunHighGrayLevelEmphasis
                end
            end
        end

        # Accumulate into global sums
        @inbounds for i in 1:num_features
            feature_sums[i] += angle_vals[i]
            if features_std
                feature_sums_sq[i] += angle_vals[i]^2
                if angle_vals[i] < feature_min[i]; feature_min[i] = angle_vals[i]; end
                if angle_vals[i] > feature_max[i]; feature_max[i] = angle_vals[i]; end
            end
        end
    end

    inv_angles     = 1.0 / num_angles
    glrlm_features = Dict{String,Any}()

    for (idx, name) in enumerate(feature_names)
        mean_val = feature_sums[idx] * inv_angles
        glrlm_features["glrlm_"*name] = mean_val

        if features_std
            # Var(X) = E[X²] - E[X]²  — clamped to 0 to avoid -epsilon from float rounding
            variance = max(feature_sums_sq[idx] * inv_angles - mean_val^2, 0.0)
            glrlm_features["glrlm_"*name*"_std"] = sqrt(variance)
            glrlm_features["glrlm_"*name*"_min"] = feature_min[idx]   # <-- aggiunto
            glrlm_features["glrlm_"*name*"_max"] = feature_max[idx] 
        end
    end

    return glrlm_features
end