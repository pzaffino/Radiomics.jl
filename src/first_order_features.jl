#=
# First order features
=#

using StatsBase


function first_order_features(img, mask, verbose=false)
    println("Extracting first order features...")

    # Some data can be useful for features extraction
    voxel_volume = img.header.pixdim[2] * img.header.pixdim[3] * img.header.pixdim[4]
    roi_voxels = extract_roi_voxels(img, mask)

    # Energy
    energy_feature_value = get_energy_feature_value(roi_voxels)
    if verbose
        println("  energy_feature_value = $energy_feature_value")
    end
    
    # Total energy
    total_energy_feature_value = get_total_energy_feature_value(voxel_volume, energy_feature_value)
    if verbose
        println("  Total energy = $total_energy_feature_value")
    end

    # Entropy
    entropy_feature_value = get_entropy_feature_value(roi_voxels)
    if verbose
        println("  Entropy = $entropy_feature_value")
    end

    # Minimum
    minimum_feature_value = get_minimum_feature_value(roi_voxels)
    if verbose
        println("  Minimum = $minimum_feature_value")
    end

    # 10th percentile
    percentile10_feature_value = get_percentile10_feature_value(roi_voxels)
    if verbose
        println("  10th percentile = $percentile10_feature_value")
    end

    # 90th percentile
    percentile90_feature_value = get_percentile90_feature_value(roi_voxels)
    if verbose
        println("  90th percentile = $percentile90_feature_value")
    end

    # Maximum
    maximum_feature_value = get_maximum_feature_value(roi_voxels)
    if verbose
        println("  Maximum = $maximum_feature_value")
    end

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

