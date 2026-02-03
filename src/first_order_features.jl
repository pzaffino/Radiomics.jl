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
function get_first_order_features(img::Array{<:Real,3}, mask::BitArray{3}, voxel_spacing::Vector{<:Real}; n_bins::Union{Int,Nothing}=nothing, bin_width::Union{Float64,Nothing}=nothing, verbose::Bool=false)
    # Convert inputs to Float64 explicitly
    img_f64 = convert(Array{Float64,3}, img)
    voxel_spacing_f64 = convert(Vector{Float64}, voxel_spacing)
    
    if verbose
        println("Extracting first order features...")
    end

    first_order_features = Dict{String,Float64}()

    # Some data can be useful for features extraction
    voxel_volume::Float64 = voxel_spacing_f64[1] * voxel_spacing_f64[2] * voxel_spacing_f64[3]
    roi_voxels::Vector{Float64} = extract_roi_voxels(img_f64, mask)

    disc, n_bins_actual, gray_levels, bin_width_used = discretize_image(img, mask; n_bins=n_bins, bin_width=bin_width)
    discretized_roi_voxels::Vector{Int} = disc[mask]

    # Energy
    energy_feature_value::Float64 = get_energy_feature_value(roi_voxels)
    first_order_features["firstorder_energy"] = energy_feature_value

    # Total energy
    total_energy_feature_value::Float64 = get_total_energy_feature_value(voxel_volume, energy_feature_value)
    first_order_features["firstorder_total_energy"] = total_energy_feature_value

    # Entropy
    entropy_feature_value::Float64 = get_entropy_feature_value(discretized_roi_voxels)
    first_order_features["firstorder_entropy"] = entropy_feature_value

    # Minimum
    minimum_feature_value::Float64 = get_minimum_feature_value(roi_voxels)
    first_order_features["firstorder_minimum"] = minimum_feature_value

    # 10th percentile
    percentile10_feature_value::Float64 = get_percentile10_feature_value(roi_voxels)
    first_order_features["firstorder_percentile10"] = percentile10_feature_value

    # 90th percentile
    percentile90_feature_value::Float64 = get_percentile90_feature_value(roi_voxels)
    first_order_features["firstorder_percentile90"] = percentile90_feature_value

    # Maximum
    maximum_feature_value::Float64 = get_maximum_feature_value(roi_voxels)
    first_order_features["firstorder_maximum"] = maximum_feature_value

    # Mean
    mean_feature_value::Float64 = get_mean_feature_value(roi_voxels)
    first_order_features["firstorder_mean"] = mean_feature_value

    # Median
    median_feature_value::Float64 = get_median_feature_value(roi_voxels)
    first_order_features["firstorder_median"] = median_feature_value

    # Interquartile range
    interquartile_range_feature_value::Float64 = get_interquartile_range_feature_value(roi_voxels)
    first_order_features["firstorder_interquartile_range"] = interquartile_range_feature_value

    # Range
    range_feature_value::Float64 = get_range_feature_value(maximum_feature_value, minimum_feature_value)
    first_order_features["firstorder_range"] = range_feature_value

    # Mean absolute deviation
    mean_absolute_deviation_feature_value::Float64 = get_mean_absolute_deviation_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_mean_absolute_deviation"] = mean_absolute_deviation_feature_value

    # Robust mean absolute deviation
    robust_mean_absolute_deviation_feature_value::Float64 = get_robust_mean_absolute_deviation_feature_value(roi_voxels)
    first_order_features["firstorder_robust_mean_absolute_deviation"] = robust_mean_absolute_deviation_feature_value

    # Root mean squared
    root_mean_squared_feature_value::Float64 = get_root_mean_squared_feature_value(roi_voxels)
    first_order_features["firstorder_root_mean_squared"] = root_mean_squared_feature_value

    # Standard deviation
    standard_deviation_feature_value::Float64 = get_standard_deviation_feature_value(roi_voxels)
    first_order_features["firstorder_standard_deviation"] = standard_deviation_feature_value

    # Skewness
    skewness_feature_value::Float64 = get_skewness_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_skewness"] = skewness_feature_value

    # Kurtosis
    kurtosis_feature_value::Float64 = get_kurtosis_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_kurtosis"] = kurtosis_feature_value

    # Variance
    variance_feature_value::Float64 = get_variance_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_variance"] = variance_feature_value

    # Uniformity
    uniformity_feature_value::Float64 = get_uniformity_feature_value(discretized_roi_voxels)
    first_order_features["firstorder_uniformity"] = uniformity_feature_value

    # Return dictionary with first order features
    return first_order_features

end

"""
    Extract the voxel values from the image within the region of interest defined by the mask.
    # Arguments
    - img::Array{Float64, 3}: 3D array representing the image
    - mask::BitArray{3}: 3D binary array representing the region of interest
    # Returns
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    """
function extract_roi_voxels(img::Array{Float64,3}, mask::BitArray{3})::Vector{Float64}
    v = vec(img[mask])
    any(isnan, v) && error("Image contains NaN values")
    return v
end

"""
    Calculate the energy feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - c::Float64: Constant to be added to each voxel value before squaring (default is 0.0)
    # Returns
    - energy_feature_value::Float64: Calculated energy feature value
    """
function get_energy_feature_value(roi_voxels::Vector{Float64}, c::Float64=0.0)::Float64
    energy_feature_value::Float64 = 0.0
    for roi_voxel in roi_voxels
        energy_feature_value = energy_feature_value + (roi_voxel + c)^2
    end
    return energy_feature_value
end

"""
   Calculate the total energy feature value from the energy feature value and voxel volume.
   # Arguments
   - voxel_volume::Float64: Volume of a single voxel
   - energy_feature_value::Float64: Energy feature value
   # Returns
   - total_energy_feature_value::Float64: Calculated total energy feature value
   """
function get_total_energy_feature_value(voxel_volume::Float64, energy_feature_value::Float64)::Float64
    total_energy_feature_value::Float64 = voxel_volume * energy_feature_value
    return total_energy_feature_value
end

"""
    Calculate the entropy feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - bin_width::Float64: Width of the bins for histogram calculation (default is 25.0)
    - eps::Float64: Small constant to avoid log(0) (default is 2.2e-16)
    # Returns
    - entropy_feature_value::Float64: Calculated entropy feature value
    """
function get_entropy_feature_value(discretized_voxels::Vector{Int})::Float64
    eps = 2.2204460492503131e-16

    unique_vals = unique(discretized_voxels)
    counts = [count(==(v), discretized_voxels) for v in unique_vals]

    p = counts / sum(counts)
    entropy_value = -sum(p .* log2.(p .+ eps))

    return Float64(entropy_value)
end
"""
    Calculate the minimum feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - minimum_feature_value::Float64: Calculated minimum feature value
    """
function get_minimum_feature_value(roi_voxels::Vector{Float64})::Float64
    return minimum(roi_voxels)
end

"""
    Calculate the 10th percentile feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - percentile10_feature_value::Float64: Calculated 10th percentile feature value
    """
function get_percentile10_feature_value(roi_voxels::Vector{Float64})::Float64
    return percentile(roi_voxels, 10)
end

"""
    Calculate the 90th percentile feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - percentile90_feature_value::Float64: Calculated 90th percentile feature value
    """
function get_percentile90_feature_value(roi_voxels::Vector{Float64})::Float64
    return percentile(roi_voxels, 90)
end

"""
    Calculate the maximum feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - maximum_feature_value::Float64: Calculated maximum feature value
    """
function get_maximum_feature_value(roi_voxels::Vector{Float64})::Float64
    return maximum(roi_voxels)
end

"""
    Calculate the mean feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - mean_feature_value::Float64: Calculated mean feature value
    """
function get_mean_feature_value(roi_voxels::Vector{Float64})::Float64
    return mean(roi_voxels)
end

"""
    Calculate the median feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - median_feature_value::Float64: Calculated median feature value
    """
function get_median_feature_value(roi_voxels::Vector{Float64})::Float64
    return median(roi_voxels)
end

"""
    Calculate the interquartile range feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - interquartile_range_feature_value::Float64: Calculated interquartile range feature value
    """
function get_interquartile_range_feature_value(roi_voxels::Vector{Float64})::Float64
    return percentile(roi_voxels, 75) - percentile(roi_voxels, 25)
end

"""
    Calculate the range feature value from the maximum and minimum feature values.
    # Arguments
    - maximum_feature_value::Float64: Maximum feature value
    - minimum_feature_value::Float64: Minimum feature value
    # Returns
    - range_feature_value::Float64: Calculated range feature value
    """
function get_range_feature_value(maximum_feature_value::Float64, minimum_feature_value::Float64)::Float64
    return maximum_feature_value - minimum_feature_value
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
    absolute_deviation::Float64 = 0.0
    for roi_voxel in roi_voxels
        absolute_deviation = absolute_deviation + abs(roi_voxel - mean_feature_value)
    end

    return (1 / size(roi_voxels, 1)) * absolute_deviation
end

"""
    Calculate the robust mean absolute deviation feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - robust_mean_absolute_deviation_feature_value::Float64: Calculated robust mean absolute deviation feature value
    """
function get_robust_mean_absolute_deviation_feature_value(roi_voxels::Vector{Float64})::Float64
    perc10::Float64 = percentile(roi_voxels, 10)
    perc90::Float64 = percentile(roi_voxels, 90)

    roi_voxels_10_90 = Vector{Float64}()
    for roi_voxel in roi_voxels
        if (roi_voxel >= perc10 && roi_voxel <= perc90)
            push!(roi_voxels_10_90, roi_voxel)
        end
    end

    mean_10_90::Float64 = mean(roi_voxels_10_90)

    robust_absolute_deviation::Float64 = 0.0
    for roi_voxel_10_90 in roi_voxels_10_90
        robust_absolute_deviation = robust_absolute_deviation + abs(roi_voxel_10_90 - mean_10_90)
    end

    return (1 / size(roi_voxels_10_90, 1)) * robust_absolute_deviation
end

"""
    Calculate the root mean squared feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - c::Float64: Constant to be added to each voxel value before squaring (default is 0.0)
    # Returns
    - root_mean_squared_feature_value::Float64: Calculated root mean squared feature value
    """
function get_root_mean_squared_feature_value(roi_voxels::Vector{Float64}, c::Float64=0.0)::Float64
    squared::Float64 = 0.0
    for roi_voxel in roi_voxels
        squared = squared + (roi_voxel + c)^2
    end

    return sqrt((1 / size(roi_voxels, 1)) * squared)
end

"""
    Calculate the standard deviation feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    # Returns
    - standard_deviation_feature_value::Float64: Calculated standard deviation feature value
    """
function get_standard_deviation_feature_value(roi_voxels::Vector{Float64})::Float64
    return std(roi_voxels)
end

"""
    Calculate the skewness feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - mean_feature_value::Float64: Mean feature value
    # Returns
    - skewness_feature_value::Float64: Calculated skewness feature value
    """
function get_skewness_feature_value(roi_voxels::Vector{Float64}, mean_feature_value::Float64)::Float64
    mu3::Float64 = 0.0
    sigma3::Float64 = 0.0
    for roi_voxel in roi_voxels
        mu3 = mu3 + (roi_voxel - mean_feature_value)^3
        sigma3 = sigma3 + (roi_voxel - mean_feature_value)^2
    end

    mu3 = (1 / size(roi_voxels, 1)) * mu3
    sigma3 = sqrt((1 / size(roi_voxels, 1)) * sigma3)^3

    return mu3 / sigma3
end

"""
    Calculate the kurtosis feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - mean_feature_value::Float64: Mean feature value
    # Returns
    - kurtosis_feature_value::Float64: Calculated kurtosis feature value
    """
function get_kurtosis_feature_value(roi_voxels::Vector{Float64}, mean_feature_value::Float64)::Float64
    mu4::Float64 = 0.0
    sigma4::Float64 = 0.0
    for roi_voxel in roi_voxels
        mu4 = mu4 + (roi_voxel - mean_feature_value)^4
        sigma4 = sigma4 + (roi_voxel - mean_feature_value)^2
    end

    mu4 = (1 / size(roi_voxels, 1)) * mu4
    sigma4 = ((1 / size(roi_voxels, 1)) * sigma4)^2

    return mu4 / sigma4
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
    squared_diff::Float64 = 0.0
    for roi_voxel in roi_voxels
        squared_diff = squared_diff + (roi_voxel - mean_feature_value)^2
    end

    return (1 / size(roi_voxels, 1)) * squared_diff
end

"""
    Calculate the uniformity feature value from the voxel values within the region of interest.
    # Arguments
    - roi_voxels::Vector{Float64}: Vector containing the voxel values within the region of interest
    - bin_width::Float64: Width of the bins for histogram calculation (default is 25.0)
    # Returns
    - uniformity_feature_value::Float64: Calculated uniformity feature value
    """
function get_uniformity_feature_value(discretized_voxels::Vector{Int})::Float64
    unique_vals = unique(discretized_voxels)
    counts = [count(==(v), discretized_voxels) for v in unique_vals]

    p = counts / sum(counts)

    return sum(p .^ 2)
end