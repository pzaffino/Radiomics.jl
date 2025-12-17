using LinearAlgebra
using Statistics

""" 
    function calculate_glcm_3d(img::Array{Float32,3}, mask::BitArray{3}, spacing::Vector{Float32}; n_bins::Union{Int,Nothing}=nothing, bin_width::Union{Float32,Nothing}=nothing, verbose::Bool=false)

    Calculates the Gray Level Co-occurrence Matrix (GLCM) for a 3D image within a specified mask.
    You can specify EITHER n_bins OR bin_width, but not both.

    # Arguments
        - `img`: The input 3D image as a Float32 array.
        - `mask`: A BitArray defining the region of interest within the image.
        - `spacing`: A vector specifying the voxel spacing in each dimension.
        - `n_bins`: The number of bins for discretizing intensity values (optional).
        - `bin_width`: The width of each bin (optional).
        - `verbose`: If true, enables verbose output for debugging or detailed processing information.

    # Returns:
        - `glcm_matrices`: A vector of GLCM matrices calculated for each direction.
        - `gray_levels`: A vector of unique gray levels present in the ROI.
        - `bin_width_used`: The bin width used for discretization.
    """
function calculate_glcm_3d(img::Array{Float32,3},
    mask::BitArray{3},
    spacing::Vector{Float32};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    verbose::Bool=false)

    disc, n_levels, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    if verbose
        println("Intensity Range: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Bin width utilized: $(bin_width_used)")
        println("Effective gray level utilized: $(n_levels)")
    end

    mask_idx = findall(mask)

    dirs = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
        (1, -1, 1), (-1, 1, 1)
    ]

    sx, sy, sz = size(disc)
    Ng = length(gray_levels)

    # Pre-allocate result vector with expected size
    glcm_matrices = Vector{Matrix{Float32}}()
    sizehint!(glcm_matrices, length(dirs))

    # Pre-compute mapping from gray levels to indices
    map_bin = Dict{Int,Int}()
    sizehint!(map_bin, Ng)
    @inbounds for (i, gl) in enumerate(gray_levels)
        map_bin[Int(gl)] = i
    end

    @inbounds for (dx, dy, dz) in dirs
        G = zeros(Float32, Ng, Ng)
        for idx in mask_idx
            x, y, z = Tuple(idx)
            nx = x + dx
            ny = y + dy
            nz = z + dz
            if 1 <= nx <= sx && 1 <= ny <= sy && 1 <= nz <= sz && mask[nx, ny, nz]
                i = map_bin[disc[x, y, z]]
                j = map_bin[disc[nx, ny, nz]]
                # Symmetrize the GLCM
                G[i, j] += 1.0f0
                G[j, i] += 1.0f0
            end
        end

        # Normalize only if non-empty
        total = sum(G)
        if total > 0
            @. G /= total  # In-place broadcasting
            push!(glcm_matrices, G)
        end
    end

    return glcm_matrices, gray_levels, bin_width_used
end

"""
    function calculate_mcc(glcm::Matrix{Float32}, px, py)
    Calculates the Maximal Correlation Coefficient (MCC) from a given Gray Level Co-occurrence Matrix (GLCM) and its marginal probabilities. 
    The @inbounds macro is used for performance optimization.

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

    # Compute Q matrix - optimized
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
    function extract_glcm_features_single(glcm::Matrix{Float32}, gray_levels::Vector{Int})

    Extracts a set of Gray Level Co-occurrence Matrix (GLCM) features from a single GLCM matrix and its corresponding gray levels. 
    It's designed to compute various texture features used in radiomics analysis. 
    The function utilizes the @inbounds macro for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float32 matrix.
        - `gray_levels`: A vector of gray levels corresponding to the GLCM.
    # Returns:
        - A dictionary (Dict{String, Float32}) containing the extracted GLCM features as key-value pairs.
    """
function extract_glcm_features_single(glcm::Matrix{Float32}, gray_levels::Vector{Int})
    features = Dict{String,Float32}()
    n_levels = length(gray_levels)
    eps = Float32(2.2e-16)

    # Compute marginal probabilities
    px = vec(sum(glcm, dims=2))
    py = vec(sum(glcm, dims=1))

    # Convert gray levels once
    gray_levels_f32 = Float32.(gray_levels)

    # Compute means
    μx = sum(gray_levels_f32[i] * px[i] for i in 1:n_levels)
    μy = sum(gray_levels_f32[j] * py[j] for j in 1:n_levels)

    # Compute standard deviations
    σx = sqrt(sum((gray_levels_f32[i] - μx)^2 * px[i] for i in 1:n_levels))
    σy = sqrt(sum((gray_levels_f32[j] - μy)^2 * py[j] for j in 1:n_levels))

    # Pre-allocate marginal distributions
    p_xminusy = zeros(Float32, n_levels)
    p_xplusy = zeros(Float32, 2 * n_levels)

    # Pre-compute marginal distributions
    @inbounds for i in 1:n_levels, j in 1:n_levels
        p = glcm[i, j]
        if p > 0
            p_xminusy[abs(i - j)+1] += p
            p_xplusy[i+j] += p
        end
    end

    kValuesDiff = Float32.(0:(n_levels-1))
    kValuesSum = Float32.(2:(2*n_levels))

    # Pre-compute correlation denominator
    corr_denom = σx * σy
    use_corr = corr_denom > 0
    inv_corr_denom = use_corr ? 1.0f0 / corr_denom : 0.0f0

    # Pre-compute inverse of n_levels
    inv_n_levels = 1.0f0 / Float32(n_levels)

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

    # Main feature extraction loop - optimized
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
                idmn += p / (1.0f0 + (d * inv_n_levels)^2)
                id += p / (1.0f0 + absd)
                idn += p / (1.0f0 + absd * inv_n_levels)

                # Inverse variance
                if i != j
                    inv_var += p / d2
                end

                # Maximum probability
                max_prob = max(max_prob, p)
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

    # Difference features - optimized loops
    diff_avg = 0.0f0
    diff_entropy = 0.0f0
    @inbounds for k in 1:n_levels
        pk = p_xminusy[k]
        diff_avg += kValuesDiff[k] * pk
        if pk > 0
            diff_entropy -= pk * log2(pk + eps)
        end
    end

    diff_var = 0.0f0
    @inbounds for k in 1:n_levels
        diff_var += (kValuesDiff[k] - diff_avg)^2 * p_xminusy[k]
    end

    features["glcm_DifferenceAverage"] = diff_avg
    features["glcm_DifferenceEntropy"] = diff_entropy
    features["glcm_DifferenceVariance"] = diff_var

    # Sum features - optimized loops
    sum_avg = 0.0f0
    sum_entropy = 0.0f0
    @inbounds for k in 2:(2*n_levels)
        pk = p_xplusy[k]
        sum_avg += kValuesSum[k-1] * pk
        if pk > 0
            sum_entropy -= pk * log2(pk + eps)
        end
    end

    features["glcm_SumAverage"] = sum_avg
    features["glcm_SumEntropy"] = sum_entropy
    features["glcm_SumSquares"] = sum_squares

    # Joint average - already computed via px
    joint_average = μx  # This is equivalent to sum(gray_levels_f32[i] * px[i])
    features["glcm_JointAverage"] = joint_average

    # Marginal entropies - optimized
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
function get_glcm_features(img::Array{Float32,3}, mask::BitArray{3}, voxel_spacing::Vector{Float32};
                            n_bins::Union{Int,Nothing}=nothing,
                            bin_width::Union{Float32,Nothing}=nothing,
                            verbose::Bool=false)

The function calculates GLCM matrices in 3D, extracts texture features from each matrix, 
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
"""
function get_glcm_features(img::Array{Float32,3},
    mask::BitArray{3},
    voxel_spacing::Vector{Float32};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    verbose::Bool=false)

    if verbose
        if !isnothing(n_bins)
            println("Calculating GLCM with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("GLCM calculation with bin_width=$(bin_width)...")
        else
            println("GLCM calculation with 32 bins (default)...")
        end
    end

    glcm_matrices, gray_levels, bin_width_used = calculate_glcm_3d(img, mask, voxel_spacing;
        n_bins=n_bins,
        bin_width=bin_width,
        verbose=verbose)


    if isempty(glcm_matrices)
        return Dict{String,Float32}()
    end

    all_features = [extract_glcm_features_single(glcm, gray_levels) for glcm in glcm_matrices]

    # Get feature names and pre-allocate result dictionary
    feature_names = collect(keys(all_features[1]))
    n_features = length(feature_names)
    n_matrices = length(all_features)
    feats = Dict{String,Float32}()
    sizehint!(feats, n_features)

    # Compute mean features more efficiently
    inv_n = 1.0f0 / Float32(n_matrices)
    @inbounds for name in feature_names
        sum_val = 0.0f0
        for f in all_features
            sum_val += f[name]
        end
        feats[name] = sum_val * inv_n
    end

    verbose && println("completed! Extract $(length(feats)) features.")

    return feats
end

"""
    function get_glcm_features(img::Matrix{Float32}, mask::BitMatrix, voxel_spacing::Vector{Float32};
                            n_bins::Union{Int,Nothing}=nothing,
                            bin_width::Union{Float32,Nothing}=nothing,
                            verbose::Bool=false)            
    Wrapper function to compute GLCM features for 2D images by reshaping them into 3D format.
    This function adds a singleton third dimension to the 2D image and mask, allowing the
    use of the 3D GLCM feature extraction function.
    # Arguments
        - `img`: The input 2D image as a Float32 matrix.
        - `mask`: A BitMatrix defining the region of interest within the image.
        - `voxel_spacing`: A vector specifying the voxel spacing in each dimension (2D).
        - `n_bins`: The number of bins for discretizing intensity values (optional).
        - `bin_width`: The width of each bin (optional).
        - `verbose`: If true, enables verbose output for debugging or detailed processing information.
    # Returns:
        - `feats`: A dictionary containing the mean GLCM features across all directions.
"""
function get_glcm_features(img::Matrix{Float32},
    mask::BitMatrix,
    voxel_spacing::Vector{Float32};
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{Float32,Nothing}=nothing,
    verbose::Bool=false)


    # Converti 3d image e mask 
    img3d = reshape(img, size(img)..., 1)
    mask3d = reshape(mask, size(mask)..., 1)

    # If 2D spacing is given, add a default spacing of 1.0 for the 3rd dimension
    spacing3d = length(voxel_spacing) == 2 ? Float32[voxel_spacing..., 1.0f0] : voxel_spacing

    return get_glcm_features(img3d,
        mask3d,
        spacing3d;
        n_bins=n_bins,
        bin_width=bin_width,
        verbose=verbose)
end
