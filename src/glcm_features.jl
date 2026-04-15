using LinearAlgebra
using Statistics
""" 
    function calculate_glcm(img::Array{Float32,3}, mask::BitArray{3}, spacing::Vector{Float32}; n_bins::Union{Int,Nothing}=nothing, bin_width::Union{Float64,Nothing}=nothing, verbose::Bool=false)

    Calculates the Gray Level Co-occurrence Matrix (GLCM) for a 3D image within a specified mask.
    You can specify EITHER n_bins OR bin_width, but not both.

    # Arguments
        - `img`: The input 3D image as a Float32 array.
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
function calculate_glcm(img,
    mask,
    spacing::Vector{Float32};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    verbose::Bool=false)

    disc, n_levels, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    dim = ndims(disc)
    dirs = if dim == 2
        [(1, 0), (0, 1), (1, 1), (1, -1)]

    else
        [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
            (1, -1, 1), (-1, 1, 1)
        ]
        
    end

    if verbose

        if !isnothing(n_bins)
            println("Calculating GLCM ($(dim)D) with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("Calculating GLCM ($(dim)D) with bin_width=$(bin_width)...")
        else
            println("Calculating GLCM ($(dim)D) with default bin_width=25...")
        end

        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Effective gray level utilized: $(n_levels)")
        println(dim == 2 ? "2D image detected. Using $(length(dirs)) directions." :
                           "3D image detected. Using $(length(dirs)) directions.")
        if weighting_norm !== nothing
            println("Weighting norm applied: $(weighting_norm)")
        end

    end

    Ng = length(gray_levels)

    glcm_matrices = Vector{Matrix{Float32}}()
    sizehint!(glcm_matrices, length(dirs))

    # Pre-compute mapping from gray levels to indices directly into mapped_disc array
    map_bin = Dict{Int,Int}()
    sizehint!(map_bin, Ng)
    @inbounds for (i, gl) in enumerate(gray_levels)
        map_bin[Int(gl)] = i
    end

    mapped_disc = zeros(Int, size(disc))
    @inbounds for i in CartesianIndices(disc)
        if mask[i]
            mapped_disc[i] = map_bin[disc[i]]
        end
    end

    # Calculate weights if weighting is specified
    weights = ones(Float32, length(dirs))
    if !isnothing(weighting_norm) && weighting_norm != "no_weighting"
        current_spacing = Tuple(spacing[1:dim])
        @inbounds for (a_idx, dir) in enumerate(dirs)
            abs_angle_dist = abs.(Float32.(dir)) .* current_spacing

            if weighting_norm == "infinity"
                weights[a_idx] = exp(-maximum(abs_angle_dist)^2)
            elseif weighting_norm == "euclidean"
                weights[a_idx] = exp(-sum(x^2 for x in abs_angle_dist))
            elseif weighting_norm == "manhattan"
                weights[a_idx] = exp(-sum(abs_angle_dist)^2)
            else
                @warn "Weighting norm \"$(weighting_norm)\" is unknown, weight set to 1"
                weights[a_idx] = 1.0f0
            end
        end

        if verbose
            println("Weights computed: ", weights)
        end
    end

    @inbounds for (dir_idx, dir) in enumerate(dirs)
        """
        CartesianIndex represents a multidimensional index in Julia.
        By converting the direction (tuple) into a CartesianIndex, we can directly
        sum the current voxel index (idx) with the direction (c_dir) to obtain
        the neighboring voxel (nidx), without extracting individual coordinates.
        This works automatically for both 2D and 3D images.
        E.g: idx = CartesianIndex(3,4,2) + CartesianIndex(1,0,0) = CartesianIndex(4,4,2)
        """
        c_dir = CartesianIndex(dir)
        G = zeros(Float32, Ng, Ng)

        for idx in CartesianIndices(disc)
            if mask[idx]
                nidx = idx + c_dir
                if checkbounds(Bool, disc, nidx) && mask[nidx]
                    i = mapped_disc[idx]
                    j = mapped_disc[nidx]
                    # Symmetrize the GLCM
                    G[i, j] += 1.0f0
                    G[j, i] += 1.0f0
                end
            end
        end

        # Handle normalization based on weighting
        if !isnothing(weighting_norm) && weighting_norm != "no_weighting"
            # Apply weight WITHOUT normalizing yet
            if sum(G) > 0
                @. G *= weights[dir_idx]
                push!(glcm_matrices, G)
            end
        else
            # No weighting: normalize each matrix separately
            total = sum(G)
            if total > 0
                @. G /= total
                push!(glcm_matrices, G)
            end
        end
    end

    # If weighting is applied, sum all weighted matrices and normalize ONCE
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
    function calculate_mcc(glcm::Matrix{Float32}, px, py)

    Calculates the Maximal Correlation Coefficient (MCC) from a given GLCM and its marginal
    probabilities. The @inbounds macro is used for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float32 matrix.
        - `px`: The marginal probability vector for rows.
        - `py`: The marginal probability vector for columns.

    # Returns:
        - The Maximal Correlation Coefficient as a Float32 value.
"""
function calculate_mcc(glcm::Matrix{Float32}, px, py)
    Ng = length(px)
    eps = 2.2f-16
    Ng < 2 && return 1.0f0

    Q = zeros(Float32, Ng, Ng)

    # Pre-compute inverse of py values
    inv_py = Vector{Float32}(undef, Ng)
    @inbounds for k in 1:Ng
        inv_py[k] = py[k] > eps ? 1.0f0 / py[k] : 0.0f0
    end

    # Compute Q matrix
    @inbounds for i in 1:Ng
        if px[i] > eps
            inv_pxi = 1.0f0 / px[i]
            for j in 1:Ng
                s = 0.0f0
                for k in 1:Ng
                    if py[k] > eps
                        s += glcm[i, k] * glcm[j, k] * inv_py[k]
                    end
                end
                Q[i, j] = s * inv_pxi
            end
        end
    end

    try
        eigs = eigvals(Q)
        vals = sort(real.(eigs), rev=true)
        (length(vals) >= 2 && vals[2] > 0) ? sqrt(vals[2]) : 0.0f0
    catch
        0.0f0
    end
end

"""
    function extract_glcm_features(glcm::Matrix{Float32}, gray_levels::Vector{Int})

    Extracts a set of GLCM features from a single GLCM matrix and its corresponding gray levels.
    The function utilizes the @inbounds macro for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float32 matrix.
        - `gray_levels`: A vector of gray levels corresponding to the GLCM.

    # Returns:
        - A dictionary (Dict{String, Float32}) containing the extracted GLCM features.
"""
function extract_glcm_features(glcm::Matrix{Float32}, gray_levels::Vector{Int})
    features = Dict{String,Float32}()
    n_levels = length(gray_levels)
    eps = Float32(2.2e-16)

    # Compute marginal probabilities
    px = vec(sum(glcm, dims=2))
    py = vec(sum(glcm, dims=1))

    # Convert gray levels once
    gray_levels_f32 = Float32.(gray_levels)

    # Compute means
    μx = 0.0f0
    μy = 0.0f0
    for i in 1:n_levels
        μx += gray_levels_f32[i] * px[i]
        μy += gray_levels_f32[i] * py[i]
    end

    # Compute standard deviations
    σx_sq = 0.0f0
    σy_sq = 0.0f0
    for i in 1:n_levels
        σx_sq += (gray_levels_f32[i] - μx)^2 * px[i]
        σy_sq += (gray_levels_f32[i] - μy)^2 * py[i]
    end
    σx = sqrt(σx_sq)
    σy = sqrt(σy_sq)

    # Pre-allocate marginal distributions using actual gray level values
    max_gray_level = maximum(gray_levels)
    min_gray_level = minimum(gray_levels)
    gray_level_range = Float32(max_gray_level - min_gray_level)

    p_xminusy = zeros(Float32, max_gray_level - min_gray_level + 1)
    p_xplusy = zeros(Float32, 2 * max_gray_level + 1)

    # Pre-compute correlation denominator
    corr_denom = σx * σy
    use_corr = corr_denom > 0
    inv_corr_denom = use_corr ? 1.0f0 / corr_denom : 0.0f0

    # Initialize feature accumulators
    autocorr = 0.0f0
    cluster_prom = 0.0f0
    cluster_shade = 0.0f0
    cluster_tend = 0.0f0
    contrast = 0.0f0
    correlation = 0.0f0
    joint_energy = 0.0f0
    joint_entropy = 0.0f0
    idm = 0.0f0
    idmn = 0.0f0
    id = 0.0f0
    idn = 0.0f0
    inv_var = 0.0f0
    max_prob = 0.0f0
    sum_squares = 0.0f0
    diff_avg = 0.0f0  # accumulated here to avoid a second i×j pass below

    ng = Float32(max_gray_level - min_gray_level + 1)

    # Main feature extraction loop — also accumulates diff_avg and populates
    # p_xminusy/p_xplusy, so we avoid two extra O(n_levels^2) passes
    @inbounds for i in 1:n_levels
        xi = gray_levels_f32[i]
        xi_minus_μx = xi - μx

        for j in 1:n_levels
            p = glcm[i, j]
            if p > 0
                yj = gray_levels_f32[j]
                yj_minus_μy = yj - μy

                # Autocorrelation
                autocorr += xi * yj * p

                # Cluster features
                s = xi_minus_μx + yj_minus_μy
                s2 = s * s
                cluster_tend += s2 * p
                cluster_shade += s * s2 * p
                cluster_prom += s2 * s2 * p

                # Difference
                d = xi - yj
                d2 = d * d
                contrast += d2 * p

                # Correlation
                if use_corr
                    correlation += xi_minus_μx * yj_minus_μy * p * inv_corr_denom
                end

                # Energy and entropy
                joint_energy += p * p
                joint_entropy -= p * log2(p + eps)
                sum_squares += xi_minus_μx * xi_minus_μx * p

                # Inverse difference features
                absd = abs(d)
                idm += p / (1.0f0 + d2)
                id += p / (1.0f0 + absd)
                idmn += p / (1.0f0 + (absd / ng)^2)
                idn  += p / (1.0f0 + absd / ng)

                # Inverse variance
                if i != j
                    inv_var += p / d2
                end

                # Maximum probability
                max_prob = max(max_prob, p)

                # Populate marginal distributions using gray level VALUES
                diff_val = abs(gray_levels[i] - gray_levels[j])
                sum_val = gray_levels[i] + gray_levels[j]
                p_xminusy[diff_val + 1] += p
                p_xplusy[sum_val] += p

                # Accumulate diff_avg in the same pass (avoids a second i×j loop)
                diff_avg += Float32(diff_val) * p
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

    # Difference entropy from p_xminusy (single pass over p_xminusy, no extra i×j loop)
    diff_entropy = 0.0f0
    @inbounds for k in 1:length(p_xminusy)
        pk = p_xminusy[k]
        if pk > 0
            diff_entropy -= pk * log2(pk + eps)
        end
    end

    # Difference variance — requires diff_avg, which was computed in the main loop above,
    # so this is now the only remaining extra i×j pass (unavoidable without two-pass logic)
    diff_var = 0.0f0
    @inbounds for i in 1:n_levels, j in 1:n_levels
        p = glcm[i, j]
        if p > 0
            diff_val = abs(gray_levels_f32[i] - gray_levels_f32[j])
            diff_var += (diff_val - diff_avg)^2 * p
        end
    end

    features["glcm_DifferenceEntropy"] = diff_entropy
    features["glcm_DifferenceVariance"] = diff_var

    # Sum features
    min_k = 2 * min_gray_level
    max_k = 2 * max_gray_level

    sum_avg = 0.0f0
    sum_entropy = 0.0f0
    @inbounds for sum_k in min_k:max_k
        pk = p_xplusy[sum_k]
        sum_avg += Float32(sum_k) * pk
        if pk > 0
            sum_entropy -= pk * log2(pk + eps)
        end
    end

    features["glcm_SumAverage"] = sum_avg
    features["glcm_SumEntropy"] = sum_entropy
    features["glcm_SumSquares"] = sum_squares

    # Joint average
    features["glcm_JointAverage"] = μx

    # Marginal entropies
    HX = 0.0f0
    HY = 0.0f0
    @inbounds for i in 1:n_levels
        pxi = px[i]
        pyi = py[i]
        if pxi > 0
            HX -= pxi * log2(pxi + eps)
        end
        if pyi > 0
            HY -= pyi * log2(pyi + eps)
        end
    end
    HXY = joint_entropy

    # Information measures of correlation
    HXY1 = 0.0f0
    HXY2 = 0.0f0
    @inbounds for i in 1:n_levels, j in 1:n_levels
        pij = glcm[i, j]
        pxpy = px[i] * py[j]
        if pxpy > 0
            log_pxpy = log2(pxpy + eps)
            HXY2 -= pxpy * log_pxpy
            if pij > 0
                HXY1 -= pij * log_pxpy
            end
        end
    end

    div = max(HX, HY)
    features["glcm_Imc1"] = div > 0 ? (HXY - HXY1) / div : 0.0f0
    features["glcm_Imc2"] = HXY2 > HXY ? sqrt(1.0f0 - exp(-2.0f0 * (HXY2 - HXY))) : 0.0f0

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
        - `img`: The input 3D image as a Float32 array.
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
function get_glcm_features(img, mask, voxel_spacing;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float64,Nothing}=nothing,
    weighting_norm::Union{String,Nothing}=nothing,
    get_raw_matrices::Bool=false,
    verbose::Bool=false)

    # Ensure spacing is Float32 and has the right length for the image dimensionality
    spacing = convert(Vector{Float32}, voxel_spacing)

    glcm_matrices, gray_levels, bin_width_used = calculate_glcm(img, mask, spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        verbose=verbose)

    features = Dict{String, Any}()

    if isempty(glcm_matrices)
        return features
    end

    if get_raw_matrices
        if verbose
            println("=================================")
            println("GLCM Matrix Dimensions: $(size(glcm_matrices[1]))  →  $(size(glcm_matrices[1],1)) gray levels × $(size(glcm_matrices[1],2)) gray levels")
            println("Number of directional matrices saved: $(length(glcm_matrices))")
            println("GLCM saved in dictionary.")
            println("=================================")
        end
        features["raw_glcm_matrices"] = glcm_matrices
        return features
    end

    # Accumulate features directly into a single sum dictionary instead of
    # allocating N separate feature dicts and averaging after — saves N-1 Dict allocations
    n_matrices = length(glcm_matrices)
    sum_features = extract_glcm_features(glcm_matrices[1], gray_levels)

    @inbounds for k in 2:n_matrices
        f = extract_glcm_features(glcm_matrices[k], gray_levels)
        for (name, val) in f
            sum_features[name] += val
        end
    end

    inv_n = 1.0f0 / Float32(n_matrices)
    @inbounds for name in keys(sum_features)
        features[name] = sum_features[name] * inv_n
    end

    verbose && println("Completed! Extracted $(length(features)) features.")

    return features
end