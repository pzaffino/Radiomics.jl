function get_diagnosis_features(sample_rate::Float64,
                                 bin_width::Union{Float64,Nothing},
                                 voxel_spacing::Vector{Float64},
                                 total_time_real::Float64,
                                 weighting_norm::Union{String,Nothing},
                                 n_bins::Union{Int,Nothing},
                                 keep_largest_only::Bool,
                                 image_input::AbstractArray,
                                 img_to_use::AbstractArray,
                                 mask_input::AbstractArray,
                                 mask_to_use::AbstractArray)::Dict{String,Any}

    diagnosis_features = Dict{String, Any}()

    # Read version from Project.toml
    project_file = joinpath(dirname(dirname(@__FILE__)), "Project.toml")
    if isfile(project_file)
        project_data = TOML.parsefile(project_file) 
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
    diagnosis_features["diagnosis_Voxel_spacing"] = collect(voxel_spacing)
    diagnosis_features["diagnosis_Image_size"] = collect(size(image_input))
    diagnosis_features["diagnosis_Image_size_cropped"] = collect(size(img_to_use))
    diagnosis_features["diagnosis_Mask_size"] = collect(size(mask_input))
    diagnosis_features["diagnosis_Mask_size_cropped"] = collect(size(mask_to_use))
    diagnosis_features["diagnosis_Dimensionality_of_image"] = string(ndims(image_input)) * "D"
    
    #parameters of the system
    diagnosis_features["diagnosis_Number_of_threads"] = Threads.nthreads()
    diagnosis_features["diagnosis_Total_time_real"] = total_time_real

    return diagnosis_features
end
    
function print_features_diagnosis(title::String,
                                   features::Dict{String,Any};
                                   log_buffer::Union{Vector{String},Nothing}=nothing)::Nothing
    output = String[]
    
    push!(output, "\n--- $title ---")
    sorted_keys = sort(collect(keys(features)))
    for (i, k) in enumerate(sorted_keys)
        push!(output, "  $i. $(rpad(k, 35)) => $(features[k])")
    end
    push!(output, "Subtotal: $(length(features)) features")
    
    if isnothing(log_buffer)
        
        for line in output
            println(line)
        end
    else
        
        append!(log_buffer, output)
    end

    return nothing
    
end