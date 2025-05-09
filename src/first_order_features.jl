#=
# First order features
=#

using StatsBase


function get_first_order_features(img::Array{Float32, 3}, mask::BitArray{3}, voxel_spacing::Vector{Float32}, verbose::Bool=false)
    println("Extracting first order features...")

    first_order_features = Dict{String, Float32}()

    # Some data can be useful for features extraction
    voxel_volume::Float32 = voxel_spacing[1] * voxel_spacing[2] * voxel_spacing[3]
    roi_voxels::Vector{Float32} = extract_roi_voxels(img, mask)

    # Energy
    energy_feature_value::Float32 = get_energy_feature_value(roi_voxels)
    first_order_features["firstorder_energy"] = energy_feature_value
    if verbose
        println("  energy_feature_value = $energy_feature_value")
    end
    
    # Total energy
    total_energy_feature_value::Float32 = get_total_energy_feature_value(voxel_volume, energy_feature_value)
    first_order_features["firstorder_total_energy"] = total_energy_feature_value
    if verbose
        println("  Total energy = $total_energy_feature_value")
    end

    # Entropy
    entropy_feature_value::Float32 = get_entropy_feature_value(roi_voxels)
    first_order_features["firstorder_entropy"] = entropy_feature_value
    if verbose
        println("  Entropy = $entropy_feature_value")
    end

    # Minimum
    minimum_feature_value::Float32 = get_minimum_feature_value(roi_voxels)
    first_order_features["firstorder_minimum"] = minimum_feature_value
    if verbose
        println("  Minimum = $minimum_feature_value")
    end

    # 10th percentile
    percentile10_feature_value::Float32 = get_percentile10_feature_value(roi_voxels)
    first_order_features["firstorder_percentile10"] = percentile10_feature_value
    if verbose
        println("  10th percentile = $percentile10_feature_value")
    end

    # 90th percentile
    percentile90_feature_value::Float32 = get_percentile90_feature_value(roi_voxels)
    first_order_features["firstorder_percentile90"] = percentile90_feature_value
    if verbose
        println("  90th percentile = $percentile90_feature_value")
    end

    # Maximum
    maximum_feature_value::Float32 = get_maximum_feature_value(roi_voxels)
    first_order_features["firstorder_maximum"] = maximum_feature_value
    if verbose
        println("  Maximum = $maximum_feature_value")
    end

    # Mean
    mean_feature_value::Float32 = get_mean_feature_value(roi_voxels)
    first_order_features["firstorder_mean"] = mean_feature_value
    if verbose
        println("  Mean = $mean_feature_value")
    end

    # Median
    median_feature_value::Float32 = get_median_feature_value(roi_voxels)
    first_order_features["firstorder_median"] = median_feature_value
    if verbose
        println("  Median = $median_feature_value")
    end

    # Interquartile range
    interquartile_range_feature_value::Float32 = get_interquartile_range_feature_value(roi_voxels)
    first_order_features["firstorder_interquartile_range"] = interquartile_range_feature_value
    if verbose
        println("  Interquartile range = $interquartile_range_feature_value")
    end

    # Range
    range_feature_value::Float32 = get_range_feature_value(maximum_feature_value, minimum_feature_value)
    first_order_features["firstorder_range"] = range_feature_value
    if verbose
        println("  Range = $range_feature_value")
    end
   
    # Mean absolute deviation
    mean_absolute_deviation_feature_value::Float32 = get_mean_absolute_deviation_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_mean_absolute_deviation"] = mean_absolute_deviation_feature_value
    if verbose
        println("  Mean absolute deviation = $mean_absolute_deviation_feature_value")
    end

    # Robust mean absolute deviation
    robust_mean_absolute_deviation_feature_value::Float32 = get_robust_mean_absolute_deviation_feature_value(roi_voxels)
    first_order_features["firstorder_robust_mean_absolute_deviation"] = robust_mean_absolute_deviation_feature_value
    if verbose
        println("  Robust mean absolute deviation = $robust_mean_absolute_deviation_feature_value")
    end

    # Root mean squared
    root_mean_squared_feature_value::Float32 = get_root_mean_squared_feature_value(roi_voxels)
    first_order_features["firstorder_root_mean_squared"] = root_mean_squared_feature_value
    if verbose
        println("  Root mean squared = $root_mean_squared_feature_value")
    end

    # Standard deviation
    standard_deviation_feature_value::Float32 = get_standard_deviation_feature_value(roi_voxels)
    first_order_features["firstorder_standard_deviation"] = standard_deviation_feature_value
    if verbose
        println("  Standard deviation = $standard_deviation_feature_value")
    end

    # Skewness
    skewness_feature_value::Float32 = get_skewness_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_skewness"] = skewness_feature_value
    if verbose
        println("  Skewness = $skewness_feature_value")
    end

    # Kurtosis
    kurtosis_feature_value::Float32 = get_kurtosis_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_kurtosis"] = kurtosis_feature_value
    if verbose
        println("  Kurtosis = $kurtosis_feature_value")
    end

    # Variance
    variance_feature_value::Float32 = get_variance_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_variance"] = variance_feature_value
    if verbose
        println("  Variance = $variance_feature_value")
    end

    # Uniformity
    uniformity_feature_value::Float32 = get_uniformity_feature_value(roi_voxels)
    first_order_features["firstorder_uniformity"] = uniformity_feature_value
    if verbose
        println("  Uniformity = $uniformity_feature_value")
    end

    # Return dictionary with first order features
    return first_order_features

end



function extract_roi_voxels(img::Array{T,3}, mask::BitArray{3})::Vector{T} where T
    return vec(img[mask])
end

function get_energy_feature_value(roi_voxels::Vector{Float32}, c::Float32=0.0f0)::Float32
    energy_feature_value::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        energy_feature_value = energy_feature_value + (roi_voxel + c)^2
    end
    return energy_feature_value
end


function get_total_energy_feature_value(voxel_volume::Float32, energy_feature_value::Float32)::Float32
    total_energy_feature_value::Float32 = voxel_volume * energy_feature_value
    return total_energy_feature_value 
end


function get_entropy_feature_value(roi_voxels::Vector{Float32}, bin_width::Float32=25.0f0, eps::Float64=2.2e-16)::Float32
    max_val::Float32 = maximum(roi_voxels)
    min_val::Float32 = minimum(roi_voxels)
    edges = min_val:bin_width:(ceil(max_val/bin_width) * bin_width)

    h = fit(Histogram, roi_voxels, edges)
    p = h.weights / sum(h.weights)

    entropy_feature_value::Float64 = 0.0
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0f0
            entropy_feature_value += pi * log2(pi + eps)
        end
    end

    return -entropy_feature_value
end

function get_minimum_feature_value(roi_voxels::Vector{Float32})::Float32
    return minimum(roi_voxels)
end


function get_percentile10_feature_value(roi_voxels::Vector{Float32})::Float32
    return percentile(roi_voxels, 10)
end


function get_percentile90_feature_value(roi_voxels::Vector{Float32})::Float32
    return percentile(roi_voxels, 90)
end


function get_maximum_feature_value(roi_voxels::Vector{Float32})::Float32
    return maximum(roi_voxels)
end


function get_mean_feature_value(roi_voxels::Vector{Float32})::Float32
    return mean(roi_voxels)
end


function get_median_feature_value(roi_voxels::Vector{Float32})::Float32
    return median(roi_voxels)
end


function get_interquartile_range_feature_value(roi_voxels::Vector{Float32})::Float32
    return percentile(roi_voxels, 75) - percentile(roi_voxels, 25)
end


function get_range_feature_value(maximum_feature_value::Float32, minimum_feature_value::Float32)::Float32
    return maximum_feature_value - minimum_feature_value
end


function get_mean_absolute_deviation_feature_value(roi_voxels::Vector{Float32}, mean_feature_value::Float32)::Float32
    absolute_deviation::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        absolute_deviation = absolute_deviation + abs(roi_voxel - mean_feature_value)
    end

    return (1/size(roi_voxels, 1)) * absolute_deviation
end


function get_robust_mean_absolute_deviation_feature_value(roi_voxels::Vector{Float32})::Float32
    perc10::Float32 = percentile(roi_voxels, 10)
    perc90::Float32 = percentile(roi_voxels, 90)

    roi_voxels_10_90 = Vector{Float32}()
    for roi_voxel in roi_voxels
        if (roi_voxel >= perc10 && roi_voxel <= perc90)
            push!(roi_voxels_10_90, roi_voxel)
        end
    end

    mean_10_90::Float32 = mean(roi_voxels_10_90)

    robust_absolute_deviation::Float32 = 0.0f0
    for roi_voxel_10_90 in roi_voxels_10_90
        robust_absolute_deviation = robust_absolute_deviation + abs(roi_voxel_10_90 - mean_10_90)
    end

    return (1/size(roi_voxels_10_90, 1)) * robust_absolute_deviation
end


function get_root_mean_squared_feature_value(roi_voxels::Vector{Float32}, c::Float32=0.0f0)::Float32
    squared::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        squared = squared + (roi_voxel + c)^2
    end

    return sqrt((1/size(roi_voxels, 1)) * squared)
end


function get_standard_deviation_feature_value(roi_voxels::Vector{Float32})::Float32
    return std(roi_voxels)
end


function get_skewness_feature_value(roi_voxels::Vector{Float32}, mean_feature_value::Float32)::Float32
    mu3::Float32 = 0.0f0
    sigma3::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        mu3 = mu3 + (roi_voxel - mean_feature_value)^3
        sigma3 = sigma3 + (roi_voxel - mean_feature_value)^2
    end

    mu3 = (1/size(roi_voxels, 1)) * mu3
    sigma3 = sqrt((1/size(roi_voxels, 1)) * sigma3)^3

    return mu3/sigma3
end


function get_kurtosis_feature_value(roi_voxels::Vector{Float32}, mean_feature_value::Float32)::Float32
    mu4::Float32 = 0.0f0
    sigma4::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        mu4 = mu4 + (roi_voxel - mean_feature_value)^4
        sigma4 = sigma4 + (roi_voxel - mean_feature_value)^2
    end

    mu4 = (1/size(roi_voxels, 1)) * mu4
    sigma4 = ((1/size(roi_voxels, 1)) * sigma4)^2

    return mu4/sigma4
end


function get_variance_feature_value(roi_voxels::Vector{Float32}, mean_feature_value::Float32)::Float32
    squared_diff::Float32 = 0.0f0
    for roi_voxel in roi_voxels
        squared_diff = squared_diff + (roi_voxel - mean_feature_value)^2
    end

    return (1/size(roi_voxels, 1)) * squared_diff
end


function get_uniformity_feature_value(roi_voxels::Vector{Float32}, bin_width::Float32=25.0f0)::Float32
    max_val::Float32 = maximum(roi_voxels)
    min_val::Float32 = minimum(roi_voxels)
    edges = min_val:bin_width:(ceil(max_val/bin_width) * bin_width)

    h = fit(Histogram, roi_voxels, edges)
    p = h.weights / sum(h.weights)

    uniformity = 0.0
    for p_i in p
        uniformity = uniformity + p_i^2
    end

    return uniformity
end

