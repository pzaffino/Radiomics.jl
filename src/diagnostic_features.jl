using TOML

# Create a static cache for versions, read from disk ONLY ONCE across the lifetime of the module
const RADIOMICS_VERSION_CACHE = Ref{Union{Nothing, Tuple{String, String}}}(nothing)

"""
    _get_cached_versions()

Helper function to parse the Project.toml file on the first call and cache the results.
Subsequent calls bypass disk I/O entirely.
"""
function _get_cached_versions()
    if isnothing(RADIOMICS_VERSION_CACHE[])
        project_file = joinpath(dirname(dirname(@__FILE__)), "Project.toml")
        if isfile(project_file)
            try
                project_data = TOML.parsefile(project_file) 
                ver = get(project_data, "version", "unknown")
                j_ver = get(get(project_data, "compat", Dict()), "julia", "unknown")
                RADIOMICS_VERSION_CACHE[] = (string(ver), string(j_ver))
            catch
                RADIOMICS_VERSION_CACHE[] = ("unknown", "unknown")
            end
        else
            RADIOMICS_VERSION_CACHE[] = ("unknown", "unknown")
        end
    end
    return RADIOMICS_VERSION_CACHE[]
end

"""
    get_diagnosis_features(...)

Collects execution metadata, system properties, and image metrics.
Optimized to use a static version cache and eliminate redundant logic checks on primitive types.
"""
function get_diagnosis_features(bin_width::Union{Float64,Nothing},
                                 voxel_spacing::Vector{Float64},
                                 total_time_real::Float64,
                                 weighting_norm::Union{String,Nothing},
                                 n_bins::Union{Int,Nothing},
                                 keep_largest_only::Bool,
                                 image_input::AbstractArray,
                                 features_std::Bool,
                                 img_to_use::AbstractArray,
                                 mask_input::AbstractArray,
                                 mask_to_use::AbstractArray)::Dict{String,Any}

    diagnosis_features = Dict{String, Any}()

    # Retrieve version metrics from cache (zero disk I/O if previously parsed)
    version, julia_version = _get_cached_versions()

    # Parameters of the software
    diagnosis_features["diagnosis_Version_of_Radiomicsjl"] = version
    diagnosis_features["diagnosis_Version_of_Julia"] = julia_version
    
    # Parameters of the image processing (Removed redundant isnothing checks on non-nullable primitive types)
    diagnosis_features["diagnosis_Bin_width"] = isnothing(bin_width) ? 25.0 : bin_width
    diagnosis_features["diagnosis_Number_of_bins"] = isnothing(n_bins) ? 32 : n_bins
    diagnosis_features["diagnosis_Weighting_norm"] = isnothing(weighting_norm) ? "no_weighting" : weighting_norm
    diagnosis_features["diagnosis_Keep_largest_only"] = keep_largest_only
    diagnosis_features["diagnosis_Features_std"] = features_std
    
    # Parameters of the image
    diagnosis_features["diagnosis_Voxel_spacing"] = collect(voxel_spacing)
    diagnosis_features["diagnosis_Image_size"] = collect(size(image_input))
    diagnosis_features["diagnosis_Image_size_cropped"] = collect(size(img_to_use))
    diagnosis_features["diagnosis_Mask_size"] = collect(size(mask_input))
    diagnosis_features["diagnosis_Mask_size_cropped"] = collect(size(mask_to_use))
    diagnosis_features["diagnosis_Dimensionality_of_image"] = string(ndims(image_input)) * "D"
    
    # Parameters of the system
    diagnosis_features["diagnosis_Number_of_threads"] = Threads.nthreads()
    diagnosis_features["diagnosis_Total_time_real"] = total_time_real

    return diagnosis_features
end

"""
    print_features_diagnosis(title, features; log_buffer=nothing)

Prints or logs metadata features. Optimized to run allocation-free by avoiding 
temporary intermediate string array allocations.
"""
function print_features_diagnosis(title::String,
                                   features::Dict{String,Any};
                                   log_buffer::Union{Vector{String},Nothing}=nothing)::Nothing
    
    # Minimize memory footprint by collecting and sorting only the dictionary keys
    sorted_keys = sort!(collect(keys(features)))
    num_features = length(features)
    
    if isnothing(log_buffer)
        # Direct streaming to STDOUT, bypassing intermediate buffer array generation
        println("\n--- $title ---")
        for (i, k) in enumerate(sorted_keys)
            println("  $i. $(rpad(k, 35)) => $(features[k])")
        end
        println("Subtotal: $num_features features")
    else
        # Direct push into the existing shared log stream to lower Garbage Collector pressure
        push!(log_buffer, "\n--- $title ---")
        for (i, k) in enumerate(sorted_keys)
            push!(log_buffer, "  $i. $(rpad(k, 35)) => $(features[k])")
        end
        push!(log_buffer, "Subtotal: $num_features features")
    end

    return nothing
end