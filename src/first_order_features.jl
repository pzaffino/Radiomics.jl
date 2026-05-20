using StatsBase

"""
    Extract first order features from the image within the region of interest defined by the mask.
    # Arguments
    - img::Array{<:Real, 3}: 3D array representing the image (any numeric type)
    - mask::BitArray{3}: 3D binary array representing the region of interest
    - voxel_spacing::Vector{<:Real}: Vector of length 3 representing the voxel spacing
    - verbose::Bool: If true, print progress information
    # Returns
    - first_order_features::Dict{String, Float64}: Dictionary containing the extracted first order
      features
    """ 
function get_first_order_features(img::AbstractArray{Float64},
                                   mask::BitArray,
                                   voxel_spacing::Vector{Float64};
                                   n_bins::Union{Int,Nothing}=nothing,
                                   bin_width::Union{Float64,Nothing}=nothing,
                                   verbose::Bool=false)::Dict{String,Any}

    verbose && println("Extracting first order features...")

    roi_voxels = img[mask]
    any(isnan, roi_voxels) && error("Image contains NaN values")
    n = length(roi_voxels)

    voxel_volume = prod(voxel_spacing)

    sorted_voxels = sort(roi_voxels)
    vmin = sorted_voxels[1]
    vmax = sorted_voxels[end]

    function _percentile(sv::Vector{Float64}, p::Float64)
        idx = clamp(round(Int, p / 100.0 * length(sv)), 1, length(sv))
        return sv[idx]
    end

    p10 = _percentile(sorted_voxels, 10.0)
    p25 = _percentile(sorted_voxels, 25.0)
    p50 = _percentile(sorted_voxels, 50.0)
    p75 = _percentile(sorted_voxels, 75.0)
    p90 = _percentile(sorted_voxels, 90.0)

    mean_val = sum(roi_voxels) / n

    disc, n_bins_actual, gray_levels, bin_width_used = discretize_image(
        img, mask;
        n_bins=n_bins,
        bin_width=bin_width,
        vmin=vmin,
        vmax=vmax
    )
    discretized_roi_voxels = disc[mask]
    p_probs = get_voxel_probabilities(discretized_roi_voxels)

    first_order_features = Dict{String,Any}()

    energy_val = get_energy_feature_value(roi_voxels)
    first_order_features["firstorder_energy"] = energy_val
    first_order_features["firstorder_total_energy"] = get_total_energy_feature_value(voxel_volume, energy_val)

    first_order_features["firstorder_entropy"] = get_entropy_feature_value(p_probs)

    first_order_features["firstorder_minimum"] = vmin
    first_order_features["firstorder_maximum"] = vmax
    first_order_features["firstorder_mean"] = mean_val
    first_order_features["firstorder_median"] = p50
    first_order_features["firstorder_percentile10"] = p10
    first_order_features["firstorder_percentile90"] = p90
    first_order_features["firstorder_interquartile_range"] = p75 - p25
    first_order_features["firstorder_range"] = vmax - vmin

    first_order_features["firstorder_mean_absolute_deviation"] = get_mean_absolute_deviation_feature_value(roi_voxels, mean_val)
    first_order_features["firstorder_robust_mean_absolute_deviation"] = get_robust_mean_absolute_deviation_feature_value(roi_voxels, p10, p90)
    first_order_features["firstorder_root_mean_squared"] = get_root_mean_squared_feature_value(roi_voxels)

    skewness_val, kurtosis_val = get_skewness_and_kurtosis(roi_voxels, mean_val)
    first_order_features["firstorder_skewness"] = skewness_val
    first_order_features["firstorder_kurtosis"] = kurtosis_val

    first_order_features["firstorder_variance"] = get_variance_feature_value(roi_voxels, mean_val)
    first_order_features["firstorder_uniformity"] = get_uniformity_feature_value(p_probs)

    first_order_features["firstorder_standard_deviation"] = sqrt(first_order_features["firstorder_variance"] * n / (n - 1))

    return first_order_features
end

"""
    Calculate the voxel probabilities from the discretized voxel values within the region of interest.
    # Arguments
    - discretized_voxels::Vector{Int}: Vector containing the discretized voxel values within the region of interest
    # Returns
    - voxel_probabilities::Vector{Float64}: Vector containing the voxel probabilities within the region of interest
    """
function get_voxel_probabilities(discretized_voxels::Vector{Int})::Vector{Float64}
    counts = collect(values(countmap(discretized_voxels)))
    return counts ./ sum(counts)
end

"""
    Calculate the energy feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - c::Float64: Constant to be added to each voxel value (default is 0.0)
    # Returns
    - energy_feature_value::Float64: Calculated energy feature value
    """
function get_energy_feature_value(roi_voxels::Vector{Float64}, c::Float64=0.0)::Float64
    return sum(x -> (x + c)^2, roi_voxels)
end

"""
    Calculate the total energy feature value from the voxel values within the region of interest.
    # Arguments
    - voxel_volume::Float64: Volume of each voxel
    - energy_feature_value::Float64: Energy feature value
    # Returns
    - total_energy_feature_value::Float64: Calculated total energy feature value
    """
function get_total_energy_feature_value(voxel_volume::Float64, energy_feature_value::Float64)::Float64
    return voxel_volume * energy_feature_value
end

"""
    Calculate the entropy feature value from the voxel values within the region of interest.
    # Arguments
    - p::Vector{Float64}: Vector containing the voxel probabilities within the region of interest
    # Returns
    - entropy_feature_value::Float64: Calculated entropy feature value
    """
function get_entropy_feature_value(p::Vector{Float64})::Float64
    eps::Float64 = 2.2204460492503131e-16
    return -sum(p .* log2.(p .+ eps))
end

"""
    Calculate the mean absolute deviation feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - mean_feature_value::Float64: Mean feature value
    # Returns
    - mean_absolute_deviation_feature_value::Float64: Calculated mean absolute deviation feature value
    """
function get_mean_absolute_deviation_feature_value(roi_voxels::Vector{Float64}, mean_feature_value::Float64)::Float64
    return mean(x -> abs(x - mean_feature_value), roi_voxels)
end

"""
    Calculate the robust mean absolute deviation feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - p10::Float64: 10th percentile of the voxel values
    - p90::Float64: 90th percentile of the voxel values
    # Returns
    - robust_mean_absolute_deviation_feature_value::Float64: Calculated robust mean absolute deviation feature value
    """

function get_robust_mean_absolute_deviation_feature_value(roi_voxels::Vector{Float64}, p10::Float64, p90::Float64)::Float64
    sum_val = 0.0
    count = 0
    @inbounds for x in roi_voxels
        if p10 <= x <= p90
            sum_val += x
            count += 1
        end
    end
    
    count == 0 && return 0.0
    mean_10_90 = sum_val / count
    
    mad_sum = 0.0
    @inbounds for x in roi_voxels
        if p10 <= x <= p90
            mad_sum += abs(x - mean_10_90)
        end
    end
    return mad_sum / count
end

"""
    Calculate the root mean squared feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - c::Float64: Constant to be added to each voxel value (default is 0.0)
    # Returns
    - root_mean_squared_feature_value::Float64: Calculated root mean squared feature value
    """
function get_root_mean_squared_feature_value(roi_voxels::Vector{Float64}, c::Float64=0.0)::Float64
    return sqrt(sum(x -> (x + c)^2, roi_voxels) / length(roi_voxels))
end

"""
    Calculate the skewness and kurtosis feature values from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - mean_val::Float64: Mean feature value
    # Returns
    - skewness_val::Float64: Calculated skewness feature value
    - kurtosis_val::Float64: Calculated kurtosis feature value
    """
function get_skewness_and_kurtosis(roi_voxels::Vector{Float64}, mean_val::Float64)::Tuple{Float64,Float64}
    n = length(roi_voxels)
    mu2 = mu3 = mu4 = 0.0
    @inbounds for x in roi_voxels
        d  = x - mean_val
        d2 = d * d
        mu2 += d2
        mu3 += d2 * d
        mu4 += d2 * d2
    end
    mu2 /= n; mu3 /= n; mu4 /= n
    return mu3 / mu2^1.5, mu4 / mu2^2
end

"""
    Calculate the variance feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - mean_feature_value::Float64: Mean feature value
    # Returns
    - variance_feature_value::Float64: Calculated variance feature value
    """
function get_variance_feature_value(roi_voxels::Vector{Float64}, mean_feature_value::Float64)::Float64
    return sum(x -> (x - mean_feature_value)^2, roi_voxels) / length(roi_voxels)
end

"""
    Calculate the uniformity feature value from the voxel values within the region of interest.
    # Arguments
    - p::Vector{Float64}: Vector of probabilities for each unique voxel value
    # Returns
    - uniformity_feature_value::Float64: Calculated uniformity feature value
    """
function get_uniformity_feature_value(p::Vector{Float64})::Float64
    return sum(p .^ 2)
end