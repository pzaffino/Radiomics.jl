###############################################################
# GLCM Radiomics 3D
###############################################################

using LinearAlgebra
using Statistics  

###############################################################
# Discretization
###############################################################
function discretize_image_glcm(img::Array{Float32,3},
                               mask::BitArray{3},
                               bin_width::Float32=25.0f0)
    
    """
    function discretize_image_glcm(img::Array{Float32,3}, mask::BitArray{3}, bin_width::Float32=25.0f0)

    Discretizes the input image for GLCM calculation. Takes into account only the voxels within the provided mask, 
    and maps the intensity values to discrete gray levels based on the specified bin width.

    #Arguments:
        - `img`: The input image.
        - `mask`: The mask defining the region of interest.
        - `voxel_spacing`: The spacing of the voxels in the image.
        - `verbose`: If true, prints progress messages.

    # And return values:
        - `disc`: The discretized image as an array of integers.
        - `n_bins`: The number of discrete gray levels.
        - `gray_levels`: A vector containing the unique gray levels present in the ROI.

    """

    masked_indices = findall(mask)
    if isempty(masked_indices)
        return zeros(Int, size(img)), 0, Float32[]
    end

    vals = img[mask]
    vmin = minimum(vals)
    vmax = maximum(vals)

    bin_offset = Int(floor(vmin / bin_width))

    disc = zeros(Int, size(img))
    @inbounds for idx in masked_indices
        v = img[idx]
        b = Int(floor(v / bin_width)) - bin_offset + 1
        disc[idx] = b
    end

    gray_levels = sort(unique(disc[mask]))
    n_bins = length(gray_levels)

    return disc, n_bins, Float32.(gray_levels)
end

###############################################################
# 2) GLCM 3D
###############################################################
function calculate_glcm_3d(img::Array{Float32,3},
                           mask::BitArray{3},
                           spacing::Vector{Float32},
                           bin_width::Float32=25.0f0,
                           verbose::Bool=false)
    """ 
    function calculate_glcm_3d(img::Array{Float32,3}, mask::BitArray{3}, spacing::Vector{Float32}, bin_width::Float32=25.0f0, verbose::Bool=false)

    Calculates the Gray Level Co-occurrence Matrix (GLCM) for a 3D image within a specified mask using given voxel spacing and bin width. 
    The direcions for GLCM calculation are defined in 3D space.

    # Arguments
        - `img`: The input 3D image as a Float32 array.
        - `mask`: A BitArray defining the region of interest within the image.
        - `spacing`: A vector specifying the voxel spacing in each dimension.
        - `bin_width`: The width of the bins for discretizing intensity values.
        - `verbose`: If true, enables verbose output for debugging or detailed processing information.

    # Returns:
        - `glcm_matrices`: A vector of GLCM matrices calculated for each direction.
        - `gray_levels`: A vector of unique gray levels present in the ROI.

    """

    disc, n_levels, gray_levels = discretize_image_glcm(img, mask, bin_width)
    mask_idx = findall(mask)

    dirs = [
        (1,0,0),(0,1,0),(0,0,1),
        (1,1,0),(1,-1,0),(1,0,1),(1,0,-1),
        (0,1,1),(0,1,-1),(1,1,1),(1,1,-1),
        (1,-1,1),(-1,1,1)
    ]

    sx, sy, sz = size(disc)
    Ng = length(gray_levels)

    # Array per memorizzare le GLCM per ogni direzione
    glcm_matrices = Vector{Matrix{Float32}}()

    map_bin = Dict{Int,Int}()
    @inbounds for (i,gl) in enumerate(gray_levels)
        map_bin[Int(gl)] = i
    end

    # Calcola GLCM per ogni direzione separatamente
    @inbounds for (dx,dy,dz) in dirs
        G = zeros(Float32, Ng, Ng)
        for idx in mask_idx
            x,y,z = Tuple(idx)
            nx = x + dx
            ny = y + dy
            nz = z + dz
            if 1 <= nx <= sx && 1 <= ny <= sy && 1 <= nz <= sz && mask[nx,ny,nz]
                i = map_bin[disc[x,y,z]]
                j = map_bin[disc[nx,ny,nz]]
                G[i,j] += 1
                G[j,i] += 1  # Simmetria
            end
        end

        # Normalizza la GLCM
        total = sum(G)
        if total > 0
            G ./= total
            push!(glcm_matrices, G)
        end
    end

    return glcm_matrices, gray_levels
end

###############################################################
# 3) Maximal Correlation Coefficient
###############################################################
function calculate_mcc(glcm::Matrix{Float32}, px, py)

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
    Ng = length(px)
    eps = 2.2f-16
    Ng < 2 && return 1.0f0

    Q = zeros(Float32, Ng, Ng)
    @inbounds for i in 1:Ng, j in 1:Ng
        if px[i] > eps
            s = 0.0f0
            for k in 1:Ng
                if py[k] > eps
                    s += glcm[i,k] * glcm[j,k] / py[k]
                end
            end
            Q[i,j] = s / px[i]
        end
    end

    try
        eigs = eigvals(Q)
        vals = sort(real.(eigs), rev=true)
        (length(vals)>=2 && vals[2]>0) ? sqrt(vals[2]) : 0.0f0
    catch
        0.0f0
    end
end

###############################################################
# 4) Feature extraction
###############################################################
function extract_glcm_features_single(glcm::Matrix{Float32}, gray_levels::Vector{Float32})
    """
    function extract_glcm_features_single(glcm::Matrix{Float32}, gray_levels::Vector{Float32})

    Extracts a set of Gray Level Co-occurrence Matrix (GLCM) features from a single GLCM matrix and its corresponding gray levels. 
    It's designed to compute various texture features used in radiomics analysis. 
    The function utilizes the @inbounds macro for performance optimization.

    # Arguments:
        - `glcm`: The Gray Level Co-occurrence Matrix as a Float32 matrix.
        - `gray_levels`: A vector of gray levels corresponding to the GLCM.
    # Returns:
        - A dictionary (Dict{String, Float32}) containing the extracted GLCM features as key-value pairs.
    """
    features = Dict{String, Float32}()
    n_levels = length(gray_levels)
    eps = Float32(2.2e-16)

    px = vec(sum(glcm, dims=2))
    py = vec(sum(glcm, dims=1))

    # IMPORTANTE: uso i Gray Levels
    μx = sum(gray_levels[i] * px[i] for i in 1:n_levels)
    μy = sum(gray_levels[j] * py[j] for j in 1:n_levels)
    σx = sqrt(sum((gray_levels[i] - μx)^2 * px[i] for i in 1:n_levels))
    σy = sqrt(sum((gray_levels[j] - μy)^2 * py[j] for j in 1:n_levels))

    p_xminusy = zeros(Float32, n_levels)
    p_xplusy  = zeros(Float32, 2*n_levels)

    @inbounds for i in 1:n_levels, j in 1:n_levels
        p = glcm[i,j]
        if p > 0
            p_xminusy[abs(i-j)+1] += p
            p_xplusy[i+j] += p
        end
    end

    kValuesDiff = Float32.(0:(n_levels-1))
    kValuesSum  = Float32.(2:(2*n_levels))

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

    @inbounds for i in 1:n_levels, j in 1:n_levels
        p = glcm[i,j]
        if p > 0
            xi = gray_levels[i]
            yj = gray_levels[j]

            # AUTOCORRELATION: usa i gray_levels, non gli indici!
            autocorr += xi * yj * p

            s = (xi + yj - μx - μy)
            cluster_prom  += s^4 * p
            cluster_shade += s^3 * p
            cluster_tend  += s^2 * p

            d = xi - yj
            contrast += d^2 * p

            if σx > 0 && σy > 0
                correlation += (xi - μx) * (yj - μy) * p / (σx * σy)
            end

            joint_energy   += p^2
            joint_entropy  -= p * log2(p + eps)
            sum_squares += (xi - μx)^2 * p

            absd = abs(d)
            idm  += p / (1 + d^2)
            idmn += p / (1 + (d/n_levels)^2)
            id   += p / (1 + absd)
            idn  += p / (1 + absd/n_levels)
            (i!=j) && (inv_var += p / d^2)
            max_prob = max(max_prob, p)
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

    diff_avg = sum(kValuesDiff[k]*p_xminusy[k] for k in 1:n_levels)
    diff_entropy = -sum(p_xminusy[k]>0 ? p_xminusy[k]*log2(p_xminusy[k]+eps) : 0.0f0 for k in 1:n_levels)
    diff_var = sum((kValuesDiff[k]-diff_avg)^2 * p_xminusy[k] for k in 1:n_levels)

    features["glcm_DifferenceAverage"] = diff_avg
    features["glcm_DifferenceEntropy"] = diff_entropy
    features["glcm_DifferenceVariance"] = diff_var

    sum_avg = sum(kValuesSum[k-1]*p_xplusy[k] for k in 2:(2*n_levels))
    sum_entropy = -sum(p_xplusy[k]>0 ? p_xplusy[k]*log2(p_xplusy[k]+eps) : 0.0f0 for k in 2:(2*n_levels))

    features["glcm_SumAverage"] = sum_avg
    features["glcm_SumEntropy"] = sum_entropy
    features["glcm_SumSquares"] = sum_squares

    joint_average = sum(gray_levels[i] * sum(glcm[i, :]) for i in 1:n_levels)
    features["glcm_JointAverage"] = joint_average

    HX = -sum(px[i]>0 ? px[i]*log2(px[i]+eps) : 0.0f0 for i in 1:n_levels)
    HY = -sum(py[j]>0 ? py[j]*log2(py[j]+eps) : 0.0f0 for j in 1:n_levels)
    HXY = joint_entropy

    HXY1 = 0.0f0
    HXY2 = 0.0f0
    @inbounds for i in 1:n_levels, j in 1:n_levels
        pij = glcm[i,j]
        pxpy = px[i]*py[j]
        (pij>0 && pxpy>0) && (HXY1 -= pij*log2(pxpy+eps))
        pxpy>0 && (HXY2 -= pxpy*log2(pxpy+eps))
    end

    div = max(HX, HY)
    features["glcm_Imc1"] = div>0 ? (HXY-HXY1)/div : 0.0f0
    features["glcm_Imc2"] = HXY2>HXY ? sqrt(1.0f0 - exp(-2.0f0*(HXY2-HXY))) : 0.0f0

    features["glcm_Mcc"] = calculate_mcc(glcm, px, py)

    return features
end

###############################################################
# 5) Principal Function
###############################################################
function get_glcm_features(img::Array{Float32,3},
                           mask::BitArray{3},
                           voxel_spacing::Vector{Float32};
                           bin_width::Float32=25.0f0,
                           verbose::Bool=false)
    """
    function get_glcm_features(img::Array{Float32,3},mask::BitArray{3}, voxel_spacing::Vector{Float32};
                            bin_width::Float32=25.0f0,
                            verbose::Bool=false)

    The function calculates GLCM matrices in 3D, extracts texture features from each matrix, 
        and returns the mean values of all features across directions. 
        Optional parameters allow setting the bin width for discretization and enabling verbose output.

    # Arguments
        - `img`: The input 3D image as a Float32 array.
        - `mask`: A BitArray defining the region of interest within the image.
        - `spacing`: A vector specifying the voxel spacing in each dimension.
        - `bin_width`: The width of the bins for discretizing intensity values.
        - `verbose`: If true, enables verbose output for debugging or detailed processing information.
    # Returns:
        - `feats`: The features of GLCM

    """

    verbose && println("Calcolo GLCM…")

    glcm_matrices, gray_levels = calculate_glcm_3d(img, mask, voxel_spacing, bin_width, verbose)

    # Calcola features per ogni matrice e fai la media
    if isempty(glcm_matrices)
        return Dict{String, Float32}()
    end

    all_features = [extract_glcm_features_single(glcm, gray_levels) for glcm in glcm_matrices]

    # Media delle features su tutte le direzioni
    feature_names = keys(all_features[1])
    feats = Dict{String, Float32}()

    for name in feature_names
        feats[name] = mean([f[name] for f in all_features])
    end

    return feats
end