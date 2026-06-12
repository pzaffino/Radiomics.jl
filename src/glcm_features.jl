using LinearAlgebra
using Statistics
""" 
    function calculate_glcm(img::Array{Float64,3}, mask::BitArray{3}, spacing::Vector{Float64}; n_bins::Union{Int,Nothing}=nothing, bin_width::Union{Float64,Nothing}=nothing, verbose::Bool=false)

    Calculates the Gray Level Co-occurrence Matrix (GLCM) for a 3D image within a specified mask.
    You can specify EITHER n_bins OR bin_width, but not both.

    # Arguments
        - `img`: The input 3D image as a Float64 array.
        - `mask`: A BitArray defining the region of interest within the image.
        - `spacing`: A vector specifying the voxel spacing in each dimension.
        - `n_bins`: The number of bins for discretizing intensity values (optional).
        - `bin_width`: The width of each bin (optional).
        - `weighting_norm`: The norm used for weighting the GLCM (optional), Weighting method ("infinity (Chebyshev)", "euclidean", "manhattan", "no_weighting", or nothing for no weighting)
        - `verbose`: If true, enables verbose output for debugging or detailed processing information.

    # Returns:
        - `glcm_matrices`: A vector of GLCM matrices calculated for each direction.
        - `gray_levels`: A vector of unique gray levels present in the ROI.
        - `bin_width_used`: The bin width used for discretization.
"""
function calculate_glcm(img::AbstractArray{Float64},
                         mask::BitArray,
                         spacing::Vector{Float64};
                         n_bins::Union{Int,Nothing}=nothing,
                         bin_width::Union{Float64,Nothing}=nothing,
                         weighting_norm::Union{String,Nothing}=nothing,
                         verbose::Bool=false)::Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}

    disc, n_levels, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    dim = ndims(disc)
    dirs = dim == 2 ? 
        [(1, 0), (0, 1), (1, 1), (1, -1)] : 
        [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
            (1, -1, 1), (-1, 1, 1)
        ]

    if verbose
        if !isnothing(n_bins)
            println("Calculating GLCM ($(dim)D) with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("Calculating GLCM ($(dim)D) with bin_width=$(bin_width)...")
        else
            println("Calculating GLCM ($(dim)D) with default bin_width=25...")
        end

        println(dim == 2 ? "2D image detected. Using $(length(dirs)) directions." :
                           "3D image detected. Using $(length(dirs)) directions.")
        if weighting_norm !== nothing
            println("Weighting norm applied: $(weighting_norm)")
        end
    end

    Ng = length(gray_levels)
    glcm_matrices = Vector{Matrix{Float64}}()
    sizehint!(glcm_matrices, length(dirs))

    min_gl = Int(minimum(gray_levels))
    max_gl = Int(maximum(gray_levels))
    lut = zeros(Int, max_gl - min_gl + 1)

    @inbounds for (i, gl) in enumerate(gray_levels)
        lut[Int(gl) - min_gl + 1] = i
    end

    mapped_disc = zeros(Int, size(disc))
    @inbounds for i in CartesianIndices(disc)
        if mask[i]
            mapped_disc[i] = lut[disc[i] - min_gl + 1]
        end
    end

    weights = ones(Float64, length(dirs))
    if !isnothing(weighting_norm) && weighting_norm != "no_weighting"
        current_spacing = Tuple(spacing[1:dim])
        @inbounds for (a_idx, dir) in enumerate(dirs)
            abs_angle_dist = abs.(Float64.(dir)) .* current_spacing

            if weighting_norm == "infinity"
                weights[a_idx] = exp(-maximum(abs_angle_dist)^2)
            elseif weighting_norm == "euclidean"
                weights[a_idx] = exp(-sum(x^2 for x in abs_angle_dist))
            elseif weighting_norm == "manhattan"
                weights[a_idx] = exp(-sum(abs_angle_dist)^2)
            else
                @warn "Weighting norm \"$(weighting_norm)\" is unknown, weight set to 1"
                weights[a_idx] = 1.0
            end
        end
        verbose && println("Weights computed: ", weights)
    end

    mask_indices = findall(mask)
    c_dirs = [CartesianIndex(dir) for dir in dirs]
    
    # 3D tensor allocation to eliminate indirection and maximize cache
    G_all = zeros(Float64, length(dirs), Ng, Ng)

    @inbounds for idx in mask_indices
        i_val = mapped_disc[idx]
        for (dir_idx, c_dir) in enumerate(c_dirs)
            nidx = idx + c_dir
            if checkbounds(Bool, disc, nidx) && mask[nidx]
                j_val = mapped_disc[nidx]
                G_all[dir_idx, i_val, j_val] += 1.0
                G_all[dir_idx, j_val, i_val] += 1.0
            end
        end
    end

    # Normalization and decomposition into flat matrices
    @inbounds for dir_idx in 1:length(dirs)
        G = G_all[dir_idx, :, :] # Slice out
        total = sum(G)
        if total > 0
            if !isnothing(weighting_norm) && weighting_norm != "no_weighting"
                @. G *= weights[dir_idx]
            else
                @. G /= total
            end
            push!(glcm_matrices, G)
        end
    end

    if !isnothing(weighting_norm) && weighting_norm != "no_weighting" && !isempty(glcm_matrices)
        summed_glcm = sum(glcm_matrices)
        total = sum(summed_glcm)
        if total > 0
            summed_glcm ./= total
        end
        glcm_matrices = [summed_glcm]
    end

    return glcm_matrices, gray_levels, bin_width_used
end

"""
    function calculate_mcc(glcm::Matrix{Float64}, px::Vector{Float64}, py::Vector{Float64})::Float64

    Calculates the Maximal Correlation Coefficient (MCC) from a given GLCM and its marginal
    probabilities. The @inbounds macro is used for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float64 matrix.
        - `px`: The marginal probability vector for rows.
        - `py`: The marginal probability vector for columns.

    # Returns:
        - The Maximal Correlation Coefficient as a Float64 value.
"""
function calculate_mcc(glcm::Matrix{Float64}, px::Vector{Float64}, py::Vector{Float64})::Float64
    Ng = length(px)
    eps_val = 2.2e-16
    Ng < 2 && return 1.0

    inv_sqrt_p = Vector{Float64}(undef, Ng)
    @inbounds for i in 1:Ng
        inv_sqrt_p[i] = px[i] > eps_val ? 1.0 / sqrt(px[i]) : 0.0
    end

    # 'undef' avoids initialization to zero, and we only calculate the upper triangle
    S = Matrix{Float64}(undef, Ng, Ng)
    @inbounds for j in 1:Ng
        inv_sqrt_pj = inv_sqrt_p[j]
        for i in 1:j
            S[i, j] = glcm[i, j] * inv_sqrt_p[i] * inv_sqrt_pj
        end
    end

    try
        eigs = eigvals(Symmetric(S, :U)) # Exploit the upper triangle of the GLCM
        vals = sort(abs.(eigs), rev=true)
        length(vals) >= 2 ? vals[2] : 0.0
    catch
        0.0
    end
end

"""
    function extract_glcm_features(glcm::Matrix{Float64}, gray_levels::Vector{Int})::Dict{String,Float64}

    Extracts a set of GLCM features from a single GLCM matrix and its corresponding gray levels.
    The function utilizes the @inbounds macro for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float64 matrix.
        - `gray_levels`: A vector of gray levels corresponding to the GLCM.

    # Returns:
        - A dictionary (Dict{String, Float64}) containing the extracted GLCM features.
"""
function extract_glcm_features(glcm::Matrix{Float64}, gray_levels::Vector{Int})::Dict{String,Float64}
    features = Dict{String,Float64}()
    n_levels = length(gray_levels)
    eps_val = 2.2e-16

    # Exploit the symmetry of the GLCM. px and py are identical.
    px = vec(sum(glcm, dims=2))
    py = px 

    gl = Float64.(gray_levels)

    # Identical means and variances on both dimensions
    μx = 0.0
    for i in 1:n_levels
        μx += gl[i] * px[i]
    end
    μy = μx

    σx_sq = 0.0
    for i in 1:n_levels
        σx_sq += (gl[i] - μx)^2 * px[i]
    end
    σx = sqrt(σx_sq)
    σy = σx

    max_gray_level = maximum(gray_levels)
    min_gray_level = minimum(gray_levels)
    
    p_xminusy = zeros(Float64, max_gray_level - min_gray_level + 1)
    p_xplusy = zeros(Float64, 2 * max_gray_level + 1)

    corr_denom = σx_sq
    use_corr = corr_denom > 0
    inv_corr_denom = use_corr ? 1.0 / corr_denom : 0.0

    autocorr = cluster_prom = cluster_shade = cluster_tend = contrast = correlation = 0.0
    joint_energy = joint_entropy = idm = idmn = id = idn = inv_var = max_prob = sum_squares = 0.0
    diff_avg = diff_sq_avg = HXY1 = 0.0

    ng = Float64(max_gray_level - min_gray_level + 1)

    # Triangular cycle (j >= i). Reduces the total iterations by 50%.
    @inbounds @fastmath for i in 1:n_levels
        xi = gl[i]
        xi_minus_μx = xi - μx

        # 1. Elements on the diagonal (j == i)
        p_diag = glcm[i, i]
        if p_diag > 0
            autocorr += xi * xi * p_diag
            s = 2.0 * xi_minus_μx
            s2 = s * s
            cluster_tend += s2 * p_diag
            cluster_shade += s * s2 * p_diag
            cluster_prom += s2 * s2 * p_diag
            
            if use_corr
                correlation += xi_minus_μx * xi_minus_μx * p_diag * inv_corr_denom
            end
            
            joint_energy += p_diag * p_diag
            joint_entropy -= p_diag * log2(p_diag + eps_val)
            sum_squares += xi_minus_μx * xi_minus_μx * p_diag
            
            # d = 0, so the divisors collapse to 1.0
            idm += p_diag
            id += p_diag
            idmn += p_diag
            idn += p_diag
            
            max_prob = max(max_prob, p_diag)
            p_xminusy[1] += p_diag
            
            sum_val = gray_levels[i] + gray_levels[i]
            p_xplusy[sum_val] += p_diag
            
            pxpy = px[i] * px[i]
            if pxpy > 0
                HXY1 -= p_diag * log2(pxpy + eps_val)
            end
        end

        # 2. Off-diagonal elements (j > i) - Multiplied by 2.0 due to symmetry
        for j in (i+1):n_levels
            p = glcm[i, j]
            if p > 0
                yj = gl[j]
                yj_minus_μx = yj - μx
                
                autocorr += 2.0 * (xi * yj * p)
                
                s = xi_minus_μx + yj_minus_μx
                s2 = s * s
                cluster_tend += 2.0 * (s2 * p)
                cluster_shade += 2.0 * (s * s2 * p)
                cluster_prom += 2.0 * (s2 * s2 * p)
                
                d = xi - yj
                d2 = d * d
                absd = abs(d)
                contrast += 2.0 * (d2 * p)
                
                if use_corr
                    correlation += 2.0 * (xi_minus_μx * yj_minus_μx * p * inv_corr_denom)
                end
                
                joint_energy += 2.0 * (p * p)
                joint_entropy -= 2.0 * (p * log2(p + eps_val))
                
                sum_squares += (xi_minus_μx * xi_minus_μx + yj_minus_μx * yj_minus_μx) * p
                
                idm += 2.0 * (p / (1.0 + d2))
                id += 2.0 * (p / (1.0 + absd))
                idmn += 2.0 * (p / (1.0 + (absd / ng)^2))
                idn  += 2.0 * (p / (1.0 + absd / ng))
                
                inv_var += 2.0 * (p / d2)
                max_prob = max(max_prob, p)
                
                diff_val = abs(gray_levels[i] - gray_levels[j])
                p_xminusy[diff_val + 1] += 2.0 * p
                
                sum_val = gray_levels[i] + gray_levels[j]
                p_xplusy[sum_val] += 2.0 * p
                
                diff_avg += 2.0 * (absd * p)
                diff_sq_avg += 2.0 * (d2 * p)
                
                pxpy = px[i] * px[j]
                if pxpy > 0
                    HXY1 -= 2.0 * (p * log2(pxpy + eps_val))
                end
            end
        end
    end

    features["glcm_Autocorrelation"] = autocorr
    features["glcm_ClusterProminence"] = cluster_prom
    features["glcm_ClusterShade"] = cluster_shade
    features["glcm_ClusterTendency"] = cluster_tend
    features["glcm_Contrast"] = contrast
    features["glcm_Correlation"] = correlation
    features["glcm_JointEnergy"] = joint_energy
    features["glcm_JointEntropy"] = joint_entropy
    features["glcm_Idm"] = idm
    features["glcm_Idmn"] = idmn
    features["glcm_Id"] = id
    features["glcm_Idn"] = idn
    features["glcm_InverseVariance"] = inv_var
    features["glcm_MaximumProbability"] = max_prob
    features["glcm_DifferenceAverage"] = diff_avg

    diff_entropy = 0.0
    @inbounds for k in 1:length(p_xminusy)
        pk = p_xminusy[k]
        if pk > 0
            diff_entropy -= pk * log2(pk + eps_val)
        end
    end
    features["glcm_DifferenceEntropy"] = diff_entropy
    features["glcm_DifferenceVariance"] = diff_sq_avg - diff_avg^2

    min_k = 2 * min_gray_level
    max_k = 2 * max_gray_level
    sum_avg = sum_entropy = 0.0
    @inbounds for sum_k in min_k:max_k
        pk = p_xplusy[sum_k]
        sum_avg += Float64(sum_k) * pk
        if pk > 0
            sum_entropy -= pk * log2(pk + eps_val)
        end
    end

    features["glcm_SumAverage"] = sum_avg
    features["glcm_SumEntropy"] = sum_entropy
    features["glcm_SumSquares"] = sum_squares
    features["glcm_JointAverage"] = μx

    HX = 0.0
    @inbounds for i in 1:n_levels
        if px[i] > 0
            HX -= px[i] * log2(px[i] + eps_val)
        end
    end
    HY = HX
    HXY = joint_entropy

    HXY2 = HX + HY

    div_val = max(HX, HY)
    features["glcm_Imc1"] = div_val > 0 ? (HXY - HXY1) / div_val : 0.0
    features["glcm_Imc2"] = HXY2 > HXY ? sqrt(1.0 - exp(-2.0 * (HXY2 - HXY))) : 0.0
    features["glcm_Mcc"] = calculate_mcc(glcm, px, py)

    return features
end

"""
    get_glcm_features(img, mask, voxel_spacing; n_bins, bin_width, weighting_norm,
                      get_raw_matrices, verbose)

    Calculates GLCM matrices for a 2D or 3D image, extracts texture features from each matrix,
    and returns the mean values of all features across directions.

    You can specify EITHER n_bins (number of bins) OR bin_width (fixed bin width):
    - If n_bins is specified, bin_width is calculated automatically from the intensity range
    - If bin_width is specified, the number of bins depends on the intensity range
    - If neither is specified, defaults to n_bins=32

    # Arguments
        - `img`: The input 3D image as a Float64 array.
        - `mask`: A BitArray defining the region of interest within the image.
    - `voxel_spacing`: A vector specifying the voxel spacing in each dimension.
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `weighting_norm`: The norm used for weighting the GLCM (optional), Weighting method ("infinity (Chebyshev)", "euclidean", "manhattan", "no_weighting", or nothing for no weighting)
    - `get_raw_matrices`: If true, returns one raw (unnormalized, unweighted) GLCM matrix per direction instead of the standard aggregated result.
    - `verbose`: If true, enables verbose output for debugging or detailed processing information.
    
    # Returns:
    - `feats`: A dictionary containing the mean GLCM features across all directions.
    
    # Examples:
    # Using fixed number of bins (bin_width calculated automatically)
    features = get_glcm_features(img, mask, spacing, n_bins=64)
        
    # Using fixed bin width (number of bins calculated automatically)
    features = get_glcm_features(img, mask, spacing, bin_width=25.0f0)
        
    # Default (32 bins)
    features = get_glcm_features(img, mask, spacing)
    
    # With weighting
    features = get_glcm_features(img, mask, spacing, weighting_norm="euclidean")
"""
function get_glcm_features(img::AbstractArray{Float64},
                            mask::BitArray,
                            voxel_spacing::Vector{Float64};
                            n_bins::Union{Int,Nothing}=nothing,
                            bin_width::Union{Float64,Nothing}=nothing,
                            weighting_norm::Union{String,Nothing}=nothing,
                            features_std::Bool=false,
                            get_raw_matrices::Bool=false,
                            verbose::Bool=false)::Dict{String,Any}

    glcm_matrices, gray_levels, bin_width_used = calculate_glcm(img, mask, voxel_spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        verbose=verbose)

    if isempty(glcm_matrices)
        return Dict{String, Any}() 
    end

    if get_raw_matrices
        if verbose
            println("=================================")
            println("GLCM Matrix Dimensions: $(size(glcm_matrices[1]))")
            println("Number of directional matrices saved: $(length(glcm_matrices))")
            println("=================================")
        end

        return Dict{String, Any}("raw_glcm_matrices" => glcm_matrices)
    end

    n_matrices = length(glcm_matrices)
    inv_n = 1.0 / Float64(n_matrices)
    
    final_features = Dict{String, Any}()
    sums_sq = Dict{String, Float64}()
    mins    = Dict{String, Float64}()
    maxs    = Dict{String, Float64}()

    f1 = extract_glcm_features(glcm_matrices[1], gray_levels)
    for (name, val) in f1
        final_features[name] = val
        if features_std
            sums_sq[name] = val^2
            mins[name] = val
            maxs[name] = val
        end
    end

    @inbounds for k in 2:n_matrices
        f = extract_glcm_features(glcm_matrices[k], gray_levels)
        for (name, val) in f
            final_features[name] += val
            if features_std
                sums_sq[name] += val^2
                if val < mins[name]; mins[name] = val; end
                if val > maxs[name]; maxs[name] = val; end
            end
        end
    end

    for name in collect(keys(final_features))
        mean_val = final_features[name] * inv_n
        final_features[name] = mean_val

        if features_std
            variance = max(sums_sq[name] * inv_n - mean_val^2, 0.0)
            final_features[name*"_std"] = sqrt(variance)
            final_features[name*"_min"] = mins[name]
            final_features[name*"_max"] = maxs[name]
        end
    end

    verbose && println("Completed! Extracted $(length(final_features)) features.")

    return final_features 
end