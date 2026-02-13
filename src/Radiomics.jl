using Base.Threads

module Radiomics

include("utils.jl")
include("glcm_features.jl")
include("first_order_features.jl")
include("shape_2D_features.jl")
include("shape_3D_features.jl")
include("glszm_features.jl")
include("ngtdm_features.jl")
include("glrlm_features.jl")
include("gldm_features.jl")
include("diagnosis.jl")

using JSON3
using Pkg

"""
    extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                              force_2d=false,
                              force_2d_dimension=1,
                              n_bins=nothing,
                              bin_width=nothing,
                              weighting_norm=nothing,
                              verbose=false,
                              keep_largest_only=true,
                              features=[],
                              labels=nothing)
    
    Extracts radiomic features from the given image and mask.
    Supports both single and multiple label extraction with parallel processing.
    
    # Parameters:
    - `img_input`: The input image (Array).
    - `mask_input`: The mask defining the region of interest (Array).
    - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
    - `features`: Array of symbols specifying which features to compute. 
                 Options: :first_order, :glcm, :shape2d, :shape3d, :glszm, :ngtdm, :glrlm, :gldm.
    - `labels`: Single label (Int), multiple labels (Vector{Int}), or nothing for default (label 1).
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `weighting_norm`: Performs weight-normalized radiomic feature extraction on the input image and mask (optional).
                        Options: "infinity", "euclidean", "manhattan", and "no_weighting".
    - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
    - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
    - `keep_largest_only`: If true, keeps only the largest connected component for 3D shape features (default: true).
    - `sample_rate`: The sample rate for feature extraction (optional).
    - `verbose`: If true, prints progress messages.
        
    # Returns:
    - Single label or nothing: Dict{String,Any} with feature names as keys
    - Multiple labels: Dict{Int,Dict{String,Any}} where outer keys are label values
"""
function extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
    features=Symbol[],
    labels=nothing,
    n_bins=nothing,
    bin_width=nothing,
    weighting_norm=nothing,
    force_2d::Bool=false,
    force_2d_dimension::Int=1,
    keep_largest_only::Bool=true,
    sample_rate=0.03,
    verbose::Bool=false)

    # Convert parameters to correct types
    bin_width = isnothing(bin_width) ? Float64(25.0) : Float64(bin_width)
    sample_rate = Float64(sample_rate)
    
    # Convert features (supports String, Vector{String}, Symbol, Vector{Symbol})
    if features isa String
        features = lowercase(features) == "all" ? Symbol[] : [Symbol(lowercase(features))]
    elseif features isa AbstractVector && !isempty(features) && eltype(features) <: AbstractString
        features = Symbol[Symbol(lowercase(string(f))) for f in features]
    elseif features isa AbstractVector && !isempty(features) && !(eltype(features) <: Symbol)
        # Handle case where features might be a mix or other types
        features = Symbol[Symbol(lowercase(string(f))) for f in features]
    end
    
    # Convert labels (supports Int, Vector{Int})
    if labels isa AbstractVector
        labels = Int[Int(l) for l in labels]
    elseif !isnothing(labels) && !(labels isa Int)
        labels = Int(labels)
    elseif isnothing(labels) # No label given, default is 1
        labels = Int(1)
    end
    
    # Convert spacing (supports any numeric vector/list)
    voxel_spacing_input = Float64[Float64(s) for s in voxel_spacing_input]
    
    # Convert n_bins if provided
    if !isnothing(n_bins)
        n_bins = Int(n_bins)
    end
    
    # Convert weighting_norm if provided
    if !isnothing(weighting_norm)
        weighting_norm = String(weighting_norm)
    end

    compute_all = isempty(features) || :all in features

    # Handle multi-label processing
    if labels isa Vector{Int}
        if verbose
            println("Processing multiple labels: ", labels)
            println("Active threads: ", Threads.nthreads())
        end
        
        results = Dict{Int, Dict{String,Any}}()
        skipped_labels = Int[]
        
        # Create a lock to synchronize access to shared dictionaries
        results_lock = ReentrantLock()
        
        # Parallelize label processing with log buffering
        tasks = map(labels) do label
            Threads.@spawn begin
                # Buffer for logs of this label
                log_buffer = String[]
                
                push!(log_buffer, "\n=== Processing LABEL $label ===")
                
                # Create binary mask for the current label
                mask_to_use, voxel_count = extract_and_check_mask(mask_input, label)
                # Check if label exists
                if voxel_count == 0
                    lock(results_lock) do
                        push!(skipped_labels, label)
                    end
                    return nothing
                end
                
                push!(log_buffer, "Label $label contains $voxel_count voxels")
                if compute_all
                    push!(log_buffer, "Computing ALL features")
                else
                    push!(log_buffer, "Computing selected features: $(join(string.(features), ", "))")
                end
                if !isnothing(n_bins)
                    push!(log_buffer, "Using n_bins = $n_bins")
                else
                    push!(log_buffer, "Using bin_width = $bin_width")
                end
                if sample_rate != 0.03
                    push!(log_buffer, "Using explicit sample_rate = $sample_rate")
                end
                if keep_largest_only
                    push!(log_buffer, "keep_largest_only = true (will be applied to selected label)")
                end
                
                try
                    total_start_time = time()
                    total_time_accumulated = 0.0 
                    total_bytes_accumulated = 0

                    # Compute radiomic features passing the log_buffer
                    radiomic_features, time_acc, bytes_acc = _compute_radiomics_impl(
                        img_input, mask_to_use, voxel_spacing_input, voxel_count;
                        n_bins=n_bins,
                        bin_width=bin_width,
                        weighting_norm=weighting_norm,
                        verbose=verbose,
                        sample_rate=sample_rate,
                        keep_largest_only=keep_largest_only,
                        force_2d=force_2d,
                        force_2d_dimension=force_2d_dimension,
                        compute_all=compute_all,
                        features=features,
                        log_buffer=log_buffer
                    )
                    
                    total_time_accumulated += time_acc
                    total_bytes_accumulated += bytes_acc

                    total_time_real = time() - total_start_time

                    # Add summary to buffer
                    push!(log_buffer, "\n--- Label $label Summary ---")
                    push!(log_buffer, "Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
                    push!(log_buffer, "Real time (end-to-end): $(total_time_real) sec")
                    push!(log_buffer, "Overhead: $(total_time_real - total_time_accumulated) sec")
                    push!(log_buffer, "Total memory allocated: $(total_bytes_accumulated / 1024^2) MiB")
                    
                    diagnosis_features = get_diagnosis_features(sample_rate, bin_width, voxel_spacing_input, total_time_real, 
                                            total_bytes_accumulated, weighting_norm, n_bins, keep_largest_only,
                                            img_input, mask_to_use)
                    merge!(radiomic_features, diagnosis_features)
                    
                    # Print diagnosis features to buffer
                    print_features_diagnosis("Diagnosis Features", diagnosis_features; log_buffer=log_buffer)
                    
                    push!(log_buffer, "Total features extracted: $(length(radiomic_features))")
                    push!(log_buffer, "Features extraction completed for LABEL $label")

                    # Add label identifier
                    radiomic_features["label_id"] = label
                    
                    return (label, radiomic_features, log_buffer)
                    
                catch e
                    error_msg = "Error processing label $label: $e"
                    push!(log_buffer, error_msg)
                    @warn error_msg
                    lock(results_lock) do
                        push!(skipped_labels, label)
                    end
                    return (label, nothing, log_buffer)
                end
            end
        end
        
        # Collect results from tasks and print in order
        for task in tasks
            result = fetch(task)
            if !isnothing(result)
                label, features_dict, logs = result
                
                # Print logs in order
                if verbose
                    for line in logs
                        println(line)
                    end
                end
                
                # Save results if processing was successful
                if !isnothing(features_dict)
                    results[label] = features_dict
                end
            end
        end
        
        if verbose
            println("\n======================")
            println("Multi-label processing completed")
            println("Labels requested: ", labels)
            println("Labels successfully processed: ", sort(collect(keys(results))))
            if !isempty(skipped_labels)
                println("Labels skipped (not found or error): ", sort(skipped_labels))
            end
            println("======================")
        end
        
        return results
    end

    # Single label or default processing
    if labels isa Int # Explicit single label specified by user or the default value (1)
        label = labels
        if verbose
            println("Extracting LABEL $labels")
        end
    end
        
    mask_to_use, voxel_count = extract_and_check_mask(mask_input, label)
    if voxel_count == 0
        error("Label $label not found in mask (no voxels with this value)")
    end
        
    if verbose
        println("Label $label contains $voxel_count voxels")
    end
        
    total_start_time = time()
    total_time_accumulated = 0.0 
    total_bytes_accumulated = 0

    if verbose
        println("Active threads: ", Threads.nthreads())
        println("Extracting radiomic features...")
        println("Processing LABEL: $label")
        if compute_all
            println("Computing ALL features")
        else
            println("Computing selected features: ", join(string.(features), ", "))
        end
        if !isnothing(n_bins)
            println("Using n_bins = $n_bins")
        else
            println("Using bin_width = $bin_width")
        end
        if sample_rate != 0.03
            println("Using explicit sample_rate = $sample_rate")
        end
        if keep_largest_only
            println("keep_largest_only = true (will be applied to selected label)")
        end
    end

    # Compute radiomic features using the internal implementation
    # Do NOT pass log_buffer here (uses default =nothing for direct printing)
    radiomic_features, time_acc, bytes_acc = _compute_radiomics_impl(
        img_input, mask_to_use, voxel_spacing_input, voxel_count;
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        verbose=verbose,
        sample_rate=sample_rate,
        keep_largest_only=keep_largest_only,
        force_2d=force_2d,
        force_2d_dimension=force_2d_dimension,
        compute_all=compute_all,
        features=features
    )
    
    total_time_accumulated += time_acc
    total_bytes_accumulated += bytes_acc

    total_time_real = time() - total_start_time

    if verbose
        println("\n======================")
        println("Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
        println("Real time (end-to-end): $(total_time_real) sec")
        println("Overhead: $(total_time_real - total_time_accumulated) sec")
        println("Total memory allocated: $(total_bytes_accumulated / 1024^2) MiB")
        diagnosis_features = get_diagnosis_features(sample_rate, bin_width, voxel_spacing_input, total_time_real, 
                                                    total_bytes_accumulated, weighting_norm, n_bins, keep_largest_only,
                                                    img_input, mask_to_use)
        merge!(radiomic_features, diagnosis_features)
        print_features_diagnosis("Diagnosis Features", diagnosis_features)
        println("---------------------")
        println("Total features extracted: $(length(radiomic_features))")
        println("Features extraction completed for LABEL $label")
        println("======================")
    end

    # Add label identifier
    radiomic_features["label_id"] = label

    return radiomic_features
end

"""
    _compute_radiomics_impl(img, mask, voxel_spacing; 
                           n_bins, bin_width,
                           weighting_norm, verbose, sample_rate, 
                           keep_largest_only, compute_all, features, log_buffer)
    
    Internal function that handles parallel computation of radiomic features.
    Spawns separate threads for each feature category and collects results.
    
    # Parameters:
    - `img`: Preprocessed image array
    - `mask`: Preprocessed mask array  
    - `voxel_spacing`: Voxel spacing array
    - `voxel_count`: Number of voxels in the mask
    - `n_bins`: Number of bins for discretization
    - `bin_width`: Bin width
    - `weighting_norm`: Weighting norm for features
    - `verbose`: Print progress messages
    - `sample_rate`: Sample rate for shape features
    - `keep_largest_only`: Keep only largest connected component
    - `force_2d`: If true, forces 2D feature extraction along the specified dimension.
    - `force_2d_dimension`: The dimension along which to force 2D extraction (1, 2, or 3).
    - `compute_all`: Compute all features or only selected ones
    - `features`: Vector of feature symbols to compute
    - `log_buffer`: Optional buffer for collecting log messages (for multi-label parallel processing)
    
    # Returns:
    - Tuple of (radiomic_features::Dict, total_time_accumulated::Float64, total_bytes_accumulated::Int)
"""
function _compute_radiomics_impl(img, mask, voxel_spacing, voxel_count::Int;
    n_bins=nothing,
    bin_width=nothing,
    weighting_norm=nothing,
    verbose::Bool=false,
    sample_rate=0.03,
    keep_largest_only::Bool=true,
    force_2d::Bool=false,
    force_2d_dimension::Int=1,
    compute_all::Bool=true,
    features=Symbol[],
    log_buffer=nothing)
    
    radiomic_features = Dict{String,Any}()
    total_time_accumulated = 0.0
    total_bytes_accumulated = 0
    
    # Helper function to print or buffer log messages
    function log_println(msg::String)
        if isnothing(log_buffer)
            println(msg)
        else
            push!(log_buffer, msg)
        end
    end
    
    # Sanity check
    input_sanity_check(img, mask, verbose)

    # Validate binning parameters
    if isnothing(n_bins) && voxel_count > 0
        validate_binning_parameters(img, mask, bin_width)
    end

    # Cast and prepare inputs
    img, mask, voxel_spacing = prepare_inputs(img, mask, voxel_spacing,
                                              force_2d, force_2d_dimension)

    # Launch parallel threads for 3D features
    if ndims(img) == 3
        # 3D shape features
        if compute_all || :shape3d in features 
            t_shape3d_features = Threads.@spawn @timed get_shape3d_features(
                mask, voxel_spacing; 
                verbose=verbose, 
                sample_rate=sample_rate, 
                keep_largest_only=keep_largest_only
            )
        end

        # GLCM features
        if compute_all || :glcm in features
            t_glcm_features = Threads.@spawn @timed get_glcm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=Float32(bin_width),
                weighting_norm=weighting_norm,
                verbose=verbose
            )
        end
        
        # First order features
        if compute_all || :first_order in features
            t_first_order_features = Threads.@spawn @timed get_first_order_features(
                img, mask, voxel_spacing; 
                n_bins=n_bins, 
                bin_width=bin_width, 
                verbose=verbose
            )
        end

        # GLSZM features
        if compute_all || :glszm in features
            t_glszm_features = Threads.@spawn @timed get_glszm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=Float32(bin_width),
                verbose=verbose
            )
        end

        # NGTDM features
        if compute_all || :ngtdm in features
            t_ngtdm_features = Threads.@spawn @timed get_ngtdm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=Float32(bin_width),
                verbose=verbose
            )
        end

        # GLRLM features
        if compute_all || :glrlm in features
            t_glrlm_features = Threads.@spawn @timed get_glrlm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=Float32(bin_width),
                weighting_norm=weighting_norm,
                verbose=verbose
            )
        end

        # GLDM features
        if compute_all || :gldm in features
            t_gldm_features = Threads.@spawn @timed get_gldm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=Float32(bin_width),
                verbose=verbose
            )
        end
    end

    # Launch parallel threads for 2D features
    if ndims(mask) == 2
        if compute_all || :shape2d in features
            t_shape2d_features = Threads.@spawn @timed get_shape2d_features(
                mask, voxel_spacing, verbose
            )
        end
    end

    # Collect results from 3D features
    if ndims(img) == 3
        # First order features
        if compute_all || :first_order in features
            results_first_order = fetch(t_first_order_features)
            first_order_features = results_first_order.value
            merge!(radiomic_features, first_order_features)
            total_time_accumulated += results_first_order.time
            total_bytes_accumulated += results_first_order.bytes
            if verbose
                log_println("First order: $(results_first_order.time) sec, $(results_first_order.bytes / 1024^2) MiB")
                print_features("First Order Features", first_order_features; log_buffer=log_buffer)
            end
        end

        # GLCM features
        if compute_all || :glcm in features
            results_glcm = fetch(t_glcm_features)
            glcm_features = results_glcm.value
            merge!(radiomic_features, glcm_features)
            total_time_accumulated += results_glcm.time
            total_bytes_accumulated += results_glcm.bytes
            if verbose
                log_println("GLCM: $(results_glcm.time) sec, $(results_glcm.bytes / 1024^2) MiB")
                print_features("GLCM Features", glcm_features; log_buffer=log_buffer)
            end
        end
        
        # GLSZM features
        if compute_all || :glszm in features
            results_glszm = fetch(t_glszm_features)
            glszm_features = results_glszm.value
            merge!(radiomic_features, glszm_features)
            total_time_accumulated += results_glszm.time
            total_bytes_accumulated += results_glszm.bytes
            if verbose
                log_println("GLSZM: $(results_glszm.time) sec, $(results_glszm.bytes / 1024^2) MiB")
                print_features("GLSZM Features", glszm_features; log_buffer=log_buffer)
            end
        end

        # NGTDM features
        if compute_all || :ngtdm in features
            results_ngtdm = fetch(t_ngtdm_features)
            ngtdm_features = results_ngtdm.value
            merge!(radiomic_features, ngtdm_features)
            total_time_accumulated += results_ngtdm.time
            total_bytes_accumulated += results_ngtdm.bytes
            if verbose
                log_println("NGTDM: $(results_ngtdm.time) sec, $(results_ngtdm.bytes / 1024^2) MiB")
                print_features("NGTDM Features", ngtdm_features; log_buffer=log_buffer)
            end
        end

        # GLRLM features
        if compute_all || :glrlm in features
            results_glrlm = fetch(t_glrlm_features)
            glrlm_features = results_glrlm.value
            merge!(radiomic_features, glrlm_features)
            total_time_accumulated += results_glrlm.time
            total_bytes_accumulated += results_glrlm.bytes
            if verbose
                log_println("GLRLM: $(results_glrlm.time) sec, $(results_glrlm.bytes / 1024^2) MiB")
                print_features("GLRLM Features", glrlm_features; log_buffer=log_buffer)
            end
        end

        # GLDM features
        if compute_all || :gldm in features
            results_gldm = fetch(t_gldm_features)
            gldm_features = results_gldm.value
            merge!(radiomic_features, gldm_features)
            total_time_accumulated += results_gldm.time
            total_bytes_accumulated += results_gldm.bytes
            if verbose
                log_println("GLDM: $(results_gldm.time) sec, $(results_gldm.bytes / 1024^2) MiB")
                print_features("GLDM Features", gldm_features; log_buffer=log_buffer)
            end
        end

        # 3D shape features
        if compute_all || :shape3d in features
            results_shape3d = fetch(t_shape3d_features)
            shape_3d_features = results_shape3d.value
            merge!(radiomic_features, shape_3d_features)
            total_time_accumulated += results_shape3d.time
            total_bytes_accumulated += results_shape3d.bytes
            if verbose
                log_println("3D shape: $(results_shape3d.time) sec, $(results_shape3d.bytes / 1024^2) MiB")
                print_features("3D Shape Features", shape_3d_features; log_buffer=log_buffer)
            end
        end
    end

    # Collect results from 2D features
    if ndims(mask) == 2
        if compute_all || :shape2d in features
            results_shape2d = fetch(t_shape2d_features)
            shape_2d_features = results_shape2d.value
            merge!(radiomic_features, shape_2d_features)
            total_time_accumulated += results_shape2d.time
            total_bytes_accumulated += results_shape2d.bytes
            if verbose
                log_println("2D shape: $(results_shape2d.time) sec, $(results_shape2d.bytes / 1024^2) MiB")
                print_features("2D Shape Features", shape_2d_features; log_buffer=log_buffer)
            end
        end
    end
    
    return (radiomic_features, total_time_accumulated, total_bytes_accumulated)
end

"""
    Wrapper function to be exposed in the C shared library
"""
# Global buffer to avoid the garbage collector (for shared library)
const LAST_JSON_RESULT = Ref{String}("")

Base.@ccallable function c_extract_radiomic_features(
    img_ptr::Ptr{Float32},
    img_size_x::Int64, img_size_y::Int64, img_size_z::Int64,
    mask_ptr::Ptr{Float32},
    spacing_x::Float64, spacing_y::Float64, spacing_z::Float64,
    n_bins::Int64
)::Cstring
    global LAST_JSON_RESULT
    try
        # Prepare inputs for the main function
        dims = (Int(img_size_x), Int(img_size_y), Int(img_size_z))
        spacing = [spacing_x, spacing_y, spacing_z]
        bins_val = n_bins <= 0 ? nothing : Int(n_bins)

        img = unsafe_wrap(Array, img_ptr, dims)
        mask = unsafe_wrap(Array, mask_ptr, dims)

        # Call the main function
        c_features_dict = extract_radiomic_features(
            img, mask, spacing;
            n_bins = bins_val,
            verbose = false
        )

        # Save the features in the global buffer
        LAST_JSON_RESULT[] = JSON3.write(c_features_dict) * "\0"
        return pointer(LAST_JSON_RESULT[])

    catch e
        @error "Error during feature extraction step" exception=(e, catch_backtrace())

        err_msg = "{\"error\": \"$e\"}\0"
        LAST_JSON_RESULT[] = err_msg
        return pointer(LAST_JSON_RESULT[])
    end
end

"""
# Examples:
    # Compute all features with default label=1
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=true);
    
    # Compute only GLCM features (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:glcm], verbose=true);
    
    # Compute only first_order and shape3d (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:first_order, :shape3d], verbose=true);
    
    # Compute GLCM and GLSZM with specific bin_width (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm], 
                                        bin_width=25, 
                                        verbose=true);
    
    # Compute all texture features (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        features=[:glcm, :glszm, :glrlm, :gldm, :ngtdm], verbose=true);
    
    # Compute with sample_rate personalized (uses default label=1)
    features = Radiomics.extract_radiomic_features(img, mask, spacing; sample_rate=1.0, verbose=true)

    # Compute with keep_largest_only personalized (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, keep_largest_only=false);

    # Compute with weighting_norm personalized (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; sample_rate = 1.0, verbose = true, weighting_norm="euclidean");
    
    # Extract features for a specific label (e.g., label 2)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; labels=2, verbose=true);
    
    # Extract features for multiple labels in parallel
    results = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        labels=[1, 2, 30], 
                                        sample_rate=1.0, 
                                        verbose=true, 
                                        keep_largest_only=true);
    # Access features for label 1: results[1]
    # Access features for label 2: results[2]
    # Access features for label 30: results[30]
    
    # Note: If no label is specified, the function defaults to label=1
"""

end
