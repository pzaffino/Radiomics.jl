function get_diagnosis_features(sample_rate, bin_width, voxel_spacing, total_time_real, 
                                total_bytes_accumulated, weighting_norm, n_bins, keep_largest_only, image, mask)

    diagnosis_features = Dict{String, Any}()

    # Read version from Project.toml
    project_file = joinpath(dirname(dirname(@__FILE__)), "Project.toml")
    if isfile(project_file)
        project_data = Pkg.TOML.parsefile(project_file)
        version = get(project_data, "version", "unknown")
        julia_version = get(get(project_data, "compat", Dict()), "julia", "unknown")
    else
        version = "unknown"
        julia_version = "unknown"
    end

    #parameters of the software
    diagnosis_features["diagnosis_Version_of_Radiomicsjl"] = version
    diagnosis_features["diagnosis_Version_of_Julia"] = julia_version
    
    #parameters of the image processing
    diagnosis_features["diagnosis_Sample_rate"] = !isnothing(sample_rate) ? sample_rate : 0.03
    diagnosis_features["diagnosis_Bin_width"] = !isnothing(bin_width) ? bin_width : 25.0
    diagnosis_features["diagnosis_Number_of_bins"] = !isnothing(n_bins) ? n_bins : 32
    diagnosis_features["diagnosis_Weighting_norm"] = !isnothing(weighting_norm) ? weighting_norm : "no_weighting"
    diagnosis_features["diagnosis_Keep_largest_only"] = !isnothing(keep_largest_only) ? keep_largest_only : true
    
    #parameters of the image
    diagnosis_features["diagnosis_Voxel_spacing"] = [voxel_spacing[1], voxel_spacing[2], voxel_spacing[3]]
    diagnosis_features["diagnosis_Image_size"] = collect(size(image))
    diagnosis_features["diagnosis_Mask_size"] = collect(size(mask))
    
    #parameters of the system
    diagnosis_features["diagnosis_Number_of_threads"] = Threads.nthreads()
    diagnosis_features["diagnosis_Total_time_real"] = total_time_real
    diagnosis_features["diagnosis_Total_bytes_accumulated_MiB"] = total_bytes_accumulated / 1024^2

    return diagnosis_features
end
    
function print_features_diagnosis(title::String, features::Dict{String, Any})
    println("\n--- $title ---")
    sorted_keys = sort(collect(keys(features)))
    for (i, k) in enumerate(sorted_keys)
        println("  $i. $(rpad(k, 35)) => $(features[k])")
    end
    println("Subtotal: $(length(features)) features")
end
