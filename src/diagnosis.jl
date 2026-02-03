function get_diagnosis_features(sample_rate, bin_width, voxel_spacing)

    diagnosis_features = Dict{String, Any}()

    diagnosis_features["Sample_rate"] = !isnothing(sample_rate) ? sample_rate : 0.03
    diagnosis_features["Bin_width"] = !isnothing(bin_width) ? bin_width : 25.0
    diagnosis_features["Voxel_spacing"] = [voxel_spacing[1], voxel_spacing[2], voxel_spacing[3]]

    return diagnosis_features
end
    
function print_features_diagnosis(title::String, features::Dict{String, Any})
    println("\n--- $title ---")
    sorted_keys = sort(collect(keys(features)))
    for (i, k) in enumerate(sorted_keys)
        println("  $i. $(rpad(k, 35)) => $(features[k])")
    end
    println("Subtotal: $(length(features)) features")
    println("---------------------\n")
end
