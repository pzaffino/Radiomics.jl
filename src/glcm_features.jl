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
        println("Range intensità: [$(minimum(img[mask])), $(maximum(img[mask]))]")
        println("Bin width utilizzata: $(bin_width_used)")
        println("Numero di gray levels effettivi: $(n_levels)")
    end
    
    mask_idx = findall(mask)

    dirs = [
        (1,0,0),(0,1,0),(0,0,1),
        (1,1,0),(1,-1,0),(1,0,1),(1,0,-1),
        (0,1,1),(0,1,-1),(1,1,1),(1,1,-1),
        (1,-1,1),(-1,1,1)
    ]

    sx, sy, sz = size(disc)
    Ng = length(gray_levels)

    glcm_matrices = Vector{Matrix{Float32}}()

    map_bin = Dict{Int,Int}()
    @inbounds for (i,gl) in enumerate(gray_levels)
        map_bin[Int(gl)] = i
    end

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
                G[j,i] += 1  
            end
        end

        
        total = sum(G)
        if total > 0
            G ./= total
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
    features = Dict{String, Float32}()
    n_levels = length(gray_levels)
    eps = Float32(2.2e-16)

    px = vec(sum(glcm, dims=2))
    py = vec(sum(glcm, dims=1))

    gray_levels_f32 = Float32.(gray_levels)
    μx = sum(gray_levels_f32[i] * px[i] for i in 1:n_levels)
    μy = sum(gray_levels_f32[j] * py[j] for j in 1:n_levels)
    σx = sqrt(sum((gray_levels_f32[i] - μx)^2 * px[i] for i in 1:n_levels))
    σy = sqrt(sum((gray_levels_f32[j] - μy)^2 * py[j] for j in 1:n_levels))

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
            xi = gray_levels_f32[i]
            yj = gray_levels_f32[j]

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

    joint_average = sum(gray_levels_f32[i] * sum(glcm[i, :]) for i in 1:n_levels)
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
            println("Calcolo GLCM con $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("Calcolo GLCM con bin_width=$(bin_width)...")
        else
            println("Calcolo GLCM con 32 bins (default)...")
        end
    end

    glcm_matrices, gray_levels, bin_width_used = calculate_glcm_3d(img, mask, voxel_spacing; 
                                                                     n_bins=n_bins, 
                                                                     bin_width=bin_width, 
                                                                     verbose=verbose)


    if isempty(glcm_matrices)
        return Dict{String, Float32}()
    end

    all_features = [extract_glcm_features_single(glcm, gray_levels) for glcm in glcm_matrices]

    feature_names = keys(all_features[1])
    feats = Dict{String, Float32}()

    for name in feature_names
        feats[name] = mean([f[name] for f in all_features])
    end

    verbose && println("Completato! Estratte $(length(feats)) features.")

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
    img3d  = reshape(img, size(img)..., 1)
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
