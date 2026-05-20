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

    glcm_matrices = Vector{Matrix{Float64}}()
    sizehint!(glcm_matrices, length(dirs))

    # Pre-compute mapping using LUT
    min_gl = Int(minimum(gray_levels))
    max_gl = Int(maximum(gray_levels))
    lut = zeros(Int, max_gl - min_gl + 1)

    @inbounds for (i, gl) in enumerate(gray_levels)
        lut[Int(gl) - min_gl + 1] = i
    end

    mapped_disc = zeros(Int, size(disc))
    @inbounds for i in CartesianIndices(disc)
        if mask[i]
            #Direct access O(1) in memory.
            mapped_disc[i] = lut[disc[i] - min_gl + 1]
        end
    end

    # Calculate weights if weighting is specified
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
                weights[a_idx] = 1.0f0
            end
        end

        if verbose
            println("Weights computed: ", weights)
        end
    end

    mask_indices = findall(mask)

    """
        CartesianIndex represents a multidimensional index in Julia.
        By converting the direction (tuple) into a CartesianIndex, we can directly
        sum the current voxel index (idx) with the direction (c_dir) to obtain
        the neighboring voxel (nidx), without extracting individual coordinates.
        This works automatically for both 2D and 3D images.
        E.g: idx = CartesianIndex(3,4,2) + CartesianIndex(1,0,0) = CartesianIndex(4,4,2)
    """

    # Pre-allocating all the GLCMs
    c_dirs = [CartesianIndex(dir) for dir in dirs]
    G_all = [zeros(Float64, Ng, Ng) for _ in 1:length(dirs)]

    @inbounds for idx in mask_indices
        i_val = mapped_disc[idx]
        for (dir_idx, c_dir) in enumerate(c_dirs)
            nidx = idx + c_dir
            if checkbounds(Bool, disc, nidx) && mask[nidx]
                j_val = mapped_disc[nidx]
                G_all[dir_idx][i_val, j_val] += 1.0
                G_all[dir_idx][j_val, i_val] += 1.0
            end
        end
    end

    # Normalization and weighting
    @inbounds for dir_idx in 1:length(dirs)
        G = G_all[dir_idx]
        if !isnothing(weighting_norm) && weighting_norm != "no_weighting"
            if sum(G) > 0
                @. G *= weights[dir_idx]
                push!(glcm_matrices, G)
            end
        else
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
    eps = 2.2e-16
    Ng < 2 && return 1.0

    # Create the scaling factor
    inv_sqrt_p = Vector{Float64}(undef, Ng)
    @inbounds for i in 1:Ng
        inv_sqrt_p[i] = px[i] > eps ? 1.0 / sqrt(px[i]) : 0.0
    end

    # Construct the symmetric matrix S in O(Ng^2) instead of O(Ng^3)
    S = zeros(Float64, Ng, Ng)
    @inbounds for j in 1:Ng
        inv_sqrt_pj = inv_sqrt_p[j]
        for i in 1:Ng
            S[i, j] = glcm[i, j] * inv_sqrt_p[i] * inv_sqrt_pj
        end
    end

    try
        # Inform Julia that the matrix is symmetric.
        # This activates native LA-PACK algorithms much faster.
        eigs = eigvals(Symmetric(S))

        # The eigenvalues of Q are the squares of eigs.
        # Since the MCC is the square root of the second eigenvalue of Q,
        # sqrt(λ^2) is simply the absolute value |λ|.
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
    eps = Float64(2.2e-16)

    # Compute marginal probabilities
    px = vec(sum(glcm, dims=2))
    py = vec(sum(glcm, dims=1))

    # Convert gray levels once
    gl = Float64.(gray_levels)

    # Compute means
    μx = 0.0
    μy = 0.0
    for i in 1:n_levels
        μx += gl[i] * px[i]
        μy += gl[i] * py[i]
    end

    # Compute standard deviations
    σx_sq = 0.0
    σy_sq = 0.0
    for i in 1:n_levels
        σx_sq += (gl[i] - μx)^2 * px[i]
        σy_sq += (gl[i] - μy)^2 * py[i]
    end
    σx = sqrt(σx_sq)
    σy = sqrt(σy_sq)

    # Pre-allocate marginal distributions using actual gray level values
    max_gray_level = maximum(gray_levels)
    min_gray_level = minimum(gray_levels)
    gray_level_range = Float64(max_gray_level - min_gray_level)

    p_xminusy = zeros(Float64, max_gray_level - min_gray_level + 1)
    p_xplusy = zeros(Float64, 2 * max_gray_level + 1)

    # Pre-compute correlation denominator
    corr_denom = σx * σy
    use_corr = corr_denom > 0
    inv_corr_denom = use_corr ? 1.0 / corr_denom : 0.0

    # Initialize feature accumulators
    autocorr = 0.0
    cluster_prom = 0.0
    cluster_shade = 0.0
    cluster_tend = 0.0
    contrast = 0.0
    correlation = 0.0
    joint_energy = 0.0
    joint_entropy = 0.0
    idm = 0.0
    idmn = 0.0
    id = 0.0
    idn = 0.0
    inv_var = 0.0
    max_prob = 0.0
    sum_squares = 0.0
    diff_avg = 0.0
    diff_sq_avg = 0.0
    HXY1 = 0.0
    HXY2 = 0.0

    ng = Float64(max_gray_level - min_gray_level + 1)

    @inbounds @fastmath for i in 1:n_levels
        xi = gl[i]
        xi_minus_μx = xi - μx

        for j in 1:n_levels
            p = glcm[i, j]

            # Information measures
            pxpy = px[i] * py[j]
            if pxpy > 0
                log_pxpy = log2(pxpy + eps)
                HXY2 -= pxpy * log_pxpy
                if p > 0
                    HXY1 -= p * log_pxpy
                end
            end

            if p > 0
                yj = gl[j]
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

                diff_avg += absd * p
                diff_sq_avg += (absd^2) * p
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

    # Difference entropy from p_xminusy
    diff_entropy = 0.0
    @inbounds for k in 1:length(p_xminusy)
        pk = p_xminusy[k]
        if pk > 0
            diff_entropy -= pk * log2(pk + eps)
        end
    end

    # Difference variance
    diff_var = diff_sq_avg - diff_avg^2

    features["glcm_DifferenceEntropy"] = diff_entropy
    features["glcm_DifferenceVariance"] = diff_var

    # Sum features
    min_k = 2 * min_gray_level
    max_k = 2 * max_gray_level

    sum_avg = 0.0
    sum_entropy = 0.0
    @inbounds for sum_k in min_k:max_k
        pk = p_xplusy[sum_k]
        sum_avg += Float64(sum_k) * pk
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
    HX = 0.0
    HY = 0.0
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

    div = max(HX, HY)
    features["glcm_Imc1"] = div > 0 ? (HXY - HXY1) / div : 0.0
    features["glcm_Imc2"] = HXY2 > HXY ? sqrt(1.0f0 - exp(-2.0f0 * (HXY2 - HXY))) : 0.0

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


    f1 = extract_glcm_features(glcm_matrices[1], gray_levels)
    for (name, val) in f1
        final_features[name] = val
    end

    @inbounds for k in 2:n_matrices
        f = extract_glcm_features(glcm_matrices[k], gray_levels)
        for (name, val) in f
            final_features[name] += val
        end
    end

    for (name, val) in final_features
        final_features[name] = val * inv_n
    end

    verbose && println("Completed! Extracted $(length(final_features)) features.")

    return final_features 
end