#=
# First order features
=#

using StatsBase


function get_first_order_features(img, mask, verbose=false)
    println("Extracting first order features...")

    first_order_features = Dict() 

    # Some data can be useful for features extraction
    voxel_volume = img.header.pixdim[2] * img.header.pixdim[3] * img.header.pixdim[4]
    roi_voxels = extract_roi_voxels(img, mask)

    # Energy
    energy_feature_value = get_energy_feature_value(roi_voxels)
    first_order_features["firstorder_energy"] = energy_feature_value
    if verbose
        println("  energy_feature_value = $energy_feature_value")
    end
    
    # Total energy
    total_energy_feature_value = get_total_energy_feature_value(voxel_volume, energy_feature_value)
    first_order_features["firstorder_total_energy"] = total_energy_feature_value
    if verbose
        println("  Total energy = $total_energy_feature_value")
    end

    # Entropy
    entropy_feature_value = get_entropy_feature_value(roi_voxels)
    first_order_features["firstorder_entropy"] = entropy_feature_value
    if verbose
        println("  Entropy = $entropy_feature_value")
    end

    # Minimum
    minimum_feature_value = get_minimum_feature_value(roi_voxels)
    first_order_features["firstorder_minimum"] = minimum_feature_value
    if verbose
        println("  Minimum = $minimum_feature_value")
    end

    # 10th percentile
    percentile10_feature_value = get_percentile10_feature_value(roi_voxels)
    first_order_features["firstorder_percentile10"] = percentile10_feature_value
    if verbose
        println("  10th percentile = $percentile10_feature_value")
    end

    # 90th percentile
    percentile90_feature_value = get_percentile90_feature_value(roi_voxels)
    first_order_features["firstorder_percentile90"] = percentile90_feature_value
    if verbose
        println("  90th percentile = $percentile90_feature_value")
    end

    # Maximum
    maximum_feature_value = get_maximum_feature_value(roi_voxels)
    first_order_features["firstorder_maximum"] = maximum_feature_value
    if verbose
        println("  Maximum = $maximum_feature_value")
    end

    # Mean
    mean_feature_value = get_mean_feature_value(roi_voxels)
    first_order_features["firstorder_mean"] = mean_feature_value
    if verbose
        println("  Mean = $mean_feature_value")
    end

    # Median
    median_feature_value = get_median_feature_value(roi_voxels)
    first_order_features["firstorder_median"] = median_feature_value
    if verbose
        println("  Median = $median_feature_value")
    end

    # Interquartile range
    interquartile_range_feature_value = get_interquartile_range_feature_value(roi_voxels)
    first_order_features["firstorder_interquartile_range"] = interquartile_range_feature_value
    if verbose
        println("  Interquartile range = $interquartile_range_feature_value")
    end

    # Range
    range_feature_value = get_range_feature_value(maximum_feature_value, minimum_feature_value)
    first_order_features["firstorder_range"] = range_feature_value
    if verbose
        println("  Range = $range_feature_value")
    end
   
    # Mean absolute deviation
    mean_absolute_deviation_feature_value = get_mean_absolute_deviation_feature_value(roi_voxels, mean_feature_value)
    first_order_features["firstorder_mean_absolute_deviation"] = mean_absolute_deviation_feature_value
    if verbose
        println("  Mean absolute deviation = $mean_absolute_deviation_feature_value")
    end

    # Robust mean absolute deviation
    robust_mean_absolute_deviation_feature_value = get_robust_mean_absolute_deviation_feature_value(roi_voxels)
    first_order_features["firstorder_robust_mean_absolute_deviation"] = robust_mean_absolute_deviation_feature_value
    if verbose
        println("  Robust mean absolute deviation = $robust_mean_absolute_deviation_feature_value")
    end

    # Root mean square
    root_mean_squared_feature_value = get_root_mean_squared_feature_value(roi_voxels)
    first_order_features["firstorder_root_mean_squared"] = root_mean_squared_feature_value
    if verbose
        println("  Root mean squared = $root_mean_squared_feature_value")
    end

    # Return dictionrary with first order features
    return first_order_features

end


function extract_roi_voxels(img, mask)
    roi_voxels = Vector{Float32}()
    for i in 1:size(img, 1)
        for j in 1:size(img, 2)
            for k in 1:size(img, 3)
                if mask[i,j,k] == 1
                    push!(roi_voxels, img[i,j,k])
                end
            end
        end
    end
    return roi_voxels
end


function get_energy_feature_value(roi_voxels, c=0.0)
    energy_feature_value = 0.0
    for i in 1:size(roi_voxels, 1)
        energy_feature_value = energy_feature_value + (roi_voxels[i] + c)^2
    end
    return energy_feature_value
end


function get_total_energy_feature_value(voxel_volume, energy_feature_value)
    total_energy_feature_value = voxel_volume * energy_feature_value
    return total_energy_feature_value 
end


function get_entropy_feature_value(roi_voxels, eps=2.2e-16)

    freqs = countmap(roi_voxels)
    total = sum(values(freqs))
    probs = [roi_voxels / total for roi_voxels in values(freqs)]
    entropy_feature_value = -sum(p * log2(p + eps) for p in probs if p > 0)

    return entropy_feature_value
end

function get_minimum_feature_value(roi_voxels)
    return minimum(roi_voxels)
end


function get_percentile10_feature_value(roi_voxels)
    return percentile(roi_voxels, 10)
end


function get_percentile90_feature_value(roi_voxels)
    return percentile(roi_voxels, 90)
end


function get_maximum_feature_value(roi_voxels)
    return maximum(roi_voxels)
end


function get_mean_feature_value(roi_voxels)
    return mean(roi_voxels)
end


function get_median_feature_value(roi_voxels)
    return median(roi_voxels)
end


function get_interquartile_range_feature_value(roi_voxels)
    return percentile(roi_voxels, 75) - percentile(roi_voxels, 25)
end


function get_range_feature_value(maximum_feature_value, minimum_feature_value)
    return maximum_feature_value - minimum_feature_value
end


function get_mean_absolute_deviation_feature_value(roi_voxels, mean_feature_value)
    absolute_deviation = 0
    for i in 1:size(roi_voxels, 1)
        absolute_deviation = absolute_deviation + abs(roi_voxels[i] - mean_feature_value)
    end

    return (1/size(roi_voxels, 1)) * absolute_deviation
end


function get_robust_mean_absolute_deviation_feature_value(roi_voxels)
    perc10 = percentile(roi_voxels, 10)
    perc90 = percentile(roi_voxels, 90)

    roi_voxels_10_90 = Vector{Float32}()
    for i in 1:size(roi_voxels, 1)
        if (roi_voxels[i] >= perc10 && roi_voxels[i] <= perc90)
            push!(roi_voxels_10_90, roi_voxels[i])
        end
    end

    mean_10_90 = mean(roi_voxels_10_90)

    robust_absolute_deviation = 0
    for i in 1:size(roi_voxels_10_90, 1)
        robust_absolute_deviation = robust_absolute_deviation + abs(roi_voxels_10_90[i] - mean_10_90)
    end

    return (1/size(roi_voxels_10_90, 1)) * robust_absolute_deviation
end


function get_root_mean_squared_feature_value(roi_voxels, c=0.0)
    squared = 0
    for i in 1:size(roi_voxels, 1)
        squared = squared + (roi_voxels[i] + c)^2
    end

    return sqrt((1/size(roi_voxels, 1)) * squared)
end

