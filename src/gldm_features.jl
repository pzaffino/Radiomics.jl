using StatsBase

"""
    get_gldm_features(img, mask, voxel_spacing; n_bins=nothing, bin_width=nothing, gldm_a=0, verbose=false)

    Calculates and returns a dictionary of GLDM (Gray Level Dependence Matrix) features.

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
    - `get_raw_matrices`: If true, returns the raw GLDM matrix.
    - `gldm_a`: The alpha parameter for the GLDM calculation.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.

    # Examples
        # Using fixed number of bins (bin_width calculated automatically)
        features = get_gldm_features(img, mask, spacing, n_bins=64)
        
        # Using fixed bin width (number of bins calculated automatically)
        features = get_gldm_features(img, mask, spacing, bin_width=25.0f0)
        
        # Default (32 bins)
        features = get_gldm_features(img, mask, spacing)
"""
function get_gldm_features(img::AbstractArray{Float64},
                            mask::BitArray,
                            voxel_spacing::Vector{Float64};
                            n_bins::Union{Int,Nothing}=nothing,
                            bin_width::Union{Float64,Nothing}=nothing,
                            gldm_a::Int=0,
                            get_raw_matrices::Bool=false,
                            verbose::Bool=false)::Dict{String,Any}
    if verbose
        if !isnothing(n_bins)
            println("Calculating GLDM with $(n_bins) bins...")
        elseif !isnothing(bin_width)
            println("GLDM calculation with bin_width=$(bin_width)...")
        else
            println("GLDM calculation with 32 bins (default)...")
        end
    end

    gldm_features = Dict{String,Any}()

    discretized_img, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)

    P_gldm, gray_levels = calculate_gldm_matrix(discretized_img, mask, gldm_a, verbose)

    if get_raw_matrices
        if verbose
            println("=================================")
            println("GLDM Matrix Dimensions: $(size(P_gldm))  →  $(size(P_gldm,1)) gray levels × $(size(P_gldm,2)) dependence values")
            println("Number of zones: $(sum(P_gldm))")
            println("GLDM saved in dictionary.")
            println("=================================")
        end
        gldm_features["raw_gldm_matrix"] = P_gldm
        return gldm_features
    end

    # Extract all features in a unified pass
    extracted_features = extract_all_gldm_features(P_gldm, gray_levels)
    merge!(gldm_features, extracted_features)

    if verbose
        println("Completed! Extracted $(length(gldm_features)) features.")
    end

    return gldm_features
end

"""
    calculate_gldm_matrix(discretized_img::Array{Int},
                                mask::BitArray,
                                gldm_a::Int,
                                verbose::Bool)::Tuple{Matrix{Int}, Vector{Int}}

    Calculates the Gray Level Dependence Matrix (GLDM).

    # Arguments
    - `discretized_img`: The discretized input image.
    - `mask`: The mask defining the region of interest.
    - `gldm_a`: The alpha parameter for the GLDM calculation.
    - `verbose`: If true, prints progress messages.

    # Returns
    - A tuple containing the GLDM matrix and the gray levels present in the ROI.
"""
function calculate_gldm_matrix(discretized_img::Array{Int},
                                mask::BitArray,
                                gldm_a::Int,
                                verbose::Bool)::Tuple{Matrix{Int}, Vector{Int}}
    verbose && println("Calculating GLDM matrix...")

    masked_img  = discretized_img[mask]
    gray_levels = sort(unique(masked_img))
    num_gl      = length(gray_levels)

    # Lookup on the Array instead of using a dictionary
    min_gl = minimum(gray_levels)
    max_gl = maximum(gray_levels)
    gl_map = zeros(Int, max_gl - min_gl + 1)
    @inbounds for (i, gl) in enumerate(gray_levels)
        gl_map[gl - min_gl + 1] = i
    end

    n_dims        = ndims(discretized_img)
    max_dependence = 3^n_dims
    P_gldm        = zeros(Int, num_gl, max_dependence)

    # Offsets CartesianIndex for the neighbors
    offsets = [CartesianIndex(Tuple(o)) for o in Iterators.product((-1:1 for _ in 1:n_dims)...)
               if !all(iszero, o)]

    sz           = size(discretized_img)
    mask_indices = findall(mask)

    # Classification interior/border
    interior_mask = eltype(mask_indices)[]
    border_mask   = eltype(mask_indices)[]
    sizehint!(interior_mask, length(mask_indices))
    sizehint!(border_mask,   length(mask_indices))

    for idx in mask_indices
        t = Tuple(idx)
        if all(t[k] > 1 && t[k] < sz[k] for k in 1:n_dims)
            push!(interior_mask, idx)
        else
            push!(border_mask, idx)
        end
    end

    # Interior: no checkbounds needed
    @inbounds for idx in interior_mask
        gl     = discretized_img[idx]
        gl_idx = gl_map[gl - min_gl + 1]

        dependence_count = 1  # center always dependent
        for o in offsets
            nidx = idx + o
            if mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
                dependence_count += 1
            end
        end

        P_gldm[gl_idx, dependence_count] += 1
    end

    # Border: checkbounds needed
    for idx in border_mask
        gl     = discretized_img[idx]
        gl_idx = gl_map[gl - min_gl + 1]

        dependence_count = 1
        for o in offsets
            nidx = idx + o
            if checkbounds(Bool, discretized_img, nidx) && mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
                dependence_count += 1
            end
        end

        P_gldm[gl_idx, dependence_count] += 1
    end

    # Trim empty columns
    last_col = 0
    for j in size(P_gldm, 2):-1:1
        if any(!iszero, @view P_gldm[:, j])
            last_col = j
            break
        end
    end
    P_gldm = P_gldm[:, 1:last_col]

    return P_gldm, gray_levels
end

"""
    extract_all_gldm_features(P_gldm::Matrix{Int}, gray_levels::Vector{Int})::Dict{String, Any}

    Arguments
    - `P_gldm`: The Gray Level Dependence Matrix (GLDM).
    - `gray_levels`: The gray levels present in the ROI.

    Returns
    - A dictionary where keys are the feature names and values are the calculated feature values.
    
    Performs a consolidated single-pass calculation over the matrix profile.
    Replaces 14 disjoint feature functions to reduce iteration overhead and cache misses.
"""
function extract_all_gldm_features(P_gldm::Matrix{Int}, gray_levels::Vector{Int})::Dict{String, Any}
    Nz = sum(P_gldm)
    Nz_scaled = Nz == 0 ? 1e-6 : Float64(Nz)
    inv_Nz = 1.0 / Nz_scaled
    inv_Nz_sq = inv_Nz * inv_Nz

    num_gl = size(P_gldm, 1)
    max_dep = size(P_gldm, 2)

    # Clean 1D Vector definitions preventing multi-dimensional matrix layout allocations
    ivector = Float64.(gray_levels)
    jvector = Float64.(1:max_dep)
    ivector_sq = ivector .^ 2
    jvector_sq = jvector .^ 2

    # Linear marginal sums
    pd = vec(sum(P_gldm, dims=1))
    pg = vec(sum(P_gldm, dims=2))

    # Initialize feature metric accumulators
    f_SDE = 0.0
    f_LDE = 0.0
    f_GLNU = 0.0
    f_DNU = 0.0
    f_GLV = 0.0
    f_DV = 0.0
    f_DE = 0.0
    f_LGLE = 0.0
    f_HGLE = 0.0
    f_SDLGLE = 0.0
    f_SDHGLE = 0.0
    f_LDLGLE = 0.0
    f_LDHGLE = 0.0

    # 1D Row / Column Performance Blocks (Marginal dependence calculations)
    u_j_sum = 0.0
    @inbounds for j in 1:max_dep
        val = Float64(pd[j])
        if val > 0.0
            j_sq = jvector_sq[j]
            f_SDE += val / j_sq
            f_LDE += val * j_sq
            f_DNU += val^2
            u_j_sum += val * jvector[j]
        end
    end
    f_SDE *= inv_Nz
    f_LDE *= inv_Nz
    f_DNUN = (f_DNU * inv_Nz) * inv_Nz
    f_DNU *= inv_Nz
    u_j = u_j_sum * inv_Nz

    @inbounds for j in 1:max_dep
        val = Float64(pd[j])
        if val > 0.0
            f_DV += val * (jvector[j] - u_j)^2
        end
    end
    f_DV *= inv_Nz

    u_i_sum = 0.0
    @inbounds for i in 1:num_gl
        val = Float64(pg[i])
        if val > 0.0
            i_sq = ivector_sq[i]
            f_GLNU += val^2
            f_LGLE += val / i_sq
            f_HGLE += val * i_sq
            u_i_sum += val * ivector[i]
        end
    end
    f_GLNU *= inv_Nz
    f_LGLE *= inv_Nz
    f_HGLE *= inv_Nz
    u_i = u_i_sum * inv_Nz

    @inbounds for i in 1:num_gl
        val = Float64(pg[i])
        if val > 0.0
            f_GLV += val * (ivector[i] - u_i)^2
        end
    end
    f_GLV *= inv_Nz

    # Unified 2D loop running in Column-Major order (j outer, i inner) for maximum cache locality
    @inbounds for j in 1:max_dep
        j_sq = jvector_sq[j]
        for i in 1:num_gl
            val = Float64(P_gldm[i, j])
            if val > 0.0
                i_sq = ivector_sq[i]
                p = val * inv_Nz
                
                f_DE -= p * log2(p)
                f_SDLGLE += val / (i_sq * j_sq)
                f_SDHGLE += val * i_sq / j_sq
                f_LDLGLE += val * j_sq / i_sq
                f_LDHGLE += val * i_sq * j_sq
            end
        end
    end
    f_SDLGLE *= inv_Nz
    f_SDHGLE *= inv_Nz
    f_LDLGLE *= inv_Nz
    f_LDHGLE *= inv_Nz

    return Dict{String, Any}(
        "gldm_SmallDependenceEmphasis" => f_SDE,
        "gldm_LargeDependenceEmphasis" => f_LDE,
        "gldm_GrayLevelNonUniformity" => f_GLNU,
        "gldm_DependenceNonUniformity" => f_DNU,
        "gldm_DependenceNonUniformityNormalized" => f_DNUN,
        "gldm_GrayLevelVariance" => f_GLV,
        "gldm_DependenceVariance" => f_DV,
        "gldm_DependenceEntropy" => f_DE,
        "gldm_LowGrayLevelEmphasis" => f_LGLE,
        "gldm_HighGrayLevelEmphasis" => f_HGLE,
        "gldm_SmallDependenceLowGrayLevelEmphasis" => f_SDLGLE,
        "gldm_SmallDependenceHighGrayLevelEmphasis" => f_SDHGLE,
        "gldm_LargeDependenceLowGrayLevelEmphasis" => f_LDLGLE,
        "gldm_LargeDependenceHighGrayLevelEmphasis" => f_LDHGLE
    )
end