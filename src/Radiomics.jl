module Radiomics

using Base.Threads
using PrecompileTools
using JSON3
using TOML

include("utils/utils.jl")
include("glcm_features.jl")
include("first_order_features.jl")
include("shape_2D_features.jl")
include("shape_3D_features.jl")
include("glszm_features.jl")
include("ngtdm_features.jl")
include("glrlm_features.jl")
include("gldm_features.jl")
include("diagnostic_features.jl")

"""
    extract_radiomic_features(img_input, mask_input, voxel_spacing_input;
                              features=Symbol[],
                              labels=nothing,
                              n_bins=nothing,
                              bin_width=nothing,
                              weighting_norm=nothing,
                              keep_largest_only::Bool=true,
                              get_raw_matrices::Bool=false,
                              slices_2d=nothing,
                              verbose::Bool=false)
    
    Extracts radiomic features from the given image and mask.
    Supports both single and multiple label extraction with parallel processing (if enabled. see below).
    
    # Parameters:
    - `img_input`: The input image (Array), either 2D or 3D. If a 3D image is passed and the "slices_2d"
                            parameter is defined, it computes the features on 2D images according to the specified plan/slice pairs.
    - `mask_input`: The mask defining the region of interest (Array) with same shape of `img_input`.
    - `voxel_spacing_input`: The spacing of the voxels in the image (Array).
    - `features`: Array of symbols specifying which features to compute. 
                 Options: :first_order, :glcm, :shape2d, :shape3d, :glszm, :ngtdm, :glrlm, :gldm.
    - `labels`: Single label (Int), multiple labels (Vector{Int}), or nothing for default (label 1).
    - `n_bins`: The number of bins for discretizing intensity values (optional).
    - `bin_width`: The width of each bin (optional).
    - `weighting_norm`: Performs weight-normalized radiomic feature extraction on the input image and mask (optional).
                        Options: "infinity", "euclidean", "manhattan", and "no_weighting".
    - `keep_largest_only`: If true, keeps only the largest connected component for 3D shape features (default: true).
    - `get_raw_matrices`: If true, computes raw (unnormalized, unweighted) GLCM matrices along all directions
                           and stores them in the result as `"get_raw_matrices"` (Vector of Matrix{Float64}).
    - `slices_2d`: If present, calcule all features on 2d slice - mask, when this parameter is used you can pass 
                            a vector of tuples (plan, slice_idx) where plan is the plane number (1, 2, or 3) and slice_idx is the slice index. 
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
    keep_largest_only::Bool=true,
    get_raw_matrices::Bool=false,
    features_std::Bool=false,
    slices_2d=nothing,
    verbose::Bool=false)::Union{Dict{String,Any}, Dict{Int,Dict{String,Any}}, Dict{Tuple{Int,Int},Any}}

    # Cast all inputs to correct types
    p = _cast_inputs(
        img_input,
        mask_input,
        voxel_spacing_input,
        features,
        labels,
        n_bins,
        bin_width,
        weighting_norm,
        features_std,
        slices_2d,
        keep_largest_only,
        get_raw_matrices,
        verbose  
    )

    compute_all = isempty(p.features) || :all in p.features

    # Management slices_2d
    if !isnothing(p.slices_2d)
        results_2d = Dict{Tuple{Int,Int}, Any}()
        dict_lock  = ReentrantLock()

        Threads.@threads for i in eachindex(p.slices_2d)
            plan, slice_idx = p.slices_2d[i]

            if p.verbose
                lock(dict_lock) do
                    println("Extracting 2D slice: Plane=$plan, slice=$slice_idx")
                end
            end

            img_slice = if plan == 1
                @view p.img[slice_idx, :, :]
            elseif plan == 2
                @view p.img[:, slice_idx, :]
            elseif plan == 3
                @view p.img[:, :, slice_idx]
            else
                error("Invalid plane: $plan. Must be 1, 2, or 3")
            end

            mask_slice = if plan == 1
                @view p.mask[slice_idx, :, :]
            elseif plan == 2
                @view p.mask[:, slice_idx, :]
            elseif plan == 3
                @view p.mask[:, :, slice_idx]
            else
                error("Invalid plane: $plan. Must be 1, 2, or 3")
            end

            spacing_2d = if plan == 1
                [p.spacing[2], p.spacing[3], p.spacing[1]]
            elseif plan == 2
                [p.spacing[1], p.spacing[3], p.spacing[2]]
            elseif plan == 3
                [p.spacing[1], p.spacing[2], p.spacing[3]]
            end

            result = extract_radiomic_features(
                img_slice, mask_slice, spacing_2d;
                features          = p.features,
                labels            = p.labels,
                n_bins            = p.n_bins,
                bin_width         = p.bin_width,
                weighting_norm    = p.weighting_norm,
                features_std      = p.features_std,
                keep_largest_only = p.keep_largest_only,
                get_raw_matrices  = p.get_raw_matrices,
                verbose           = p.verbose
            )

            lock(dict_lock) do
                results_2d[(plan, slice_idx)] = result
            end
        end

        if length(p.slices_2d) == 1
            key = first(keys(results_2d))
            return results_2d[key]
        else
            return results_2d
        end
    end

    # Handle multi-label processing
    if p.labels isa Vector{Int}
        if p.verbose
            println("Processing multiple labels: ", p.labels)
            println("Active threads: ", Threads.nthreads())
        end

        results        = Dict{Int, Dict{String,Any}}()
        skipped_labels = Int[]
        results_lock   = ReentrantLock()

        tasks = map(p.labels) do label
            Threads.@spawn let label = label
                log_buffer = String[]

                push!(log_buffer, "\n=== Processing LABEL $label ===")

                local mask_to_use, voxel_count = extract_and_check_mask(p.mask, label)
                if voxel_count == 0
                    lock(results_lock) do
                        push!(skipped_labels, label)
                    end
                    return nothing
                end

                push!(log_buffer, "Label $label contains $voxel_count voxels")
                push!(log_buffer, "Applying bounding box to Label $label...")
                local img_to_use
                img_to_use, mask_to_use = bounding_box(p.img, mask_to_use, p.verbose; log_buffer=log_buffer)

                if compute_all
                    push!(log_buffer, "Computing ALL features")
                else
                    push!(log_buffer, "Computing selected features: $(join(string.(p.features), ", "))")
                end
                if !isnothing(p.n_bins)
                    push!(log_buffer, "Using n_bins = $(p.n_bins)")
                else
                    push!(log_buffer, "Using bin_width = $(p.bin_width)")
                end
                if p.keep_largest_only
                    push!(log_buffer, "keep_largest_only = true (will be applied to selected label)")
                end

                try
                    total_start_time       = time()
                    total_time_accumulated = 0.0

                    radiomic_features, time_acc = _compute_radiomics_impl(
                        img_to_use, mask_to_use, p.spacing, voxel_count;
                        n_bins            = p.n_bins,
                        bin_width         = p.bin_width,
                        weighting_norm    = p.weighting_norm,
                        verbose           = p.verbose,
                        keep_largest_only = p.keep_largest_only,
                        compute_all       = compute_all,
                        features_std      = p.features_std,
                        features          = p.features,
                        get_raw_matrices  = p.get_raw_matrices,
                        log_buffer        = log_buffer
                    )

                    total_time_accumulated += time_acc
                    total_time_real         = time() - total_start_time

                    push!(log_buffer, "\n--- Label $label Summary ---")
                    push!(log_buffer, "Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
                    push!(log_buffer, "Real time (end-to-end): $(total_time_real) sec")
                    push!(log_buffer, "Overhead: $(total_time_real - total_time_accumulated) sec")

                    diagnosis_features = get_diagnosis_features(
                        p.bin_width, p.spacing, total_time_real,
                        p.weighting_norm, p.n_bins, p.keep_largest_only,
                        p.img, p.features_std, img_to_use, p.mask, mask_to_use
                    )
                    merge!(radiomic_features, diagnosis_features)

                    print_features_diagnosis("Diagnosis Features", diagnosis_features; log_buffer=log_buffer)

                    push!(log_buffer, "Total features extracted: $(length(radiomic_features))")
                    push!(log_buffer, "Features extraction completed for LABEL $label")

                    radiomic_features["label_id"] = label

                    return (label, radiomic_features, log_buffer)

                catch e
                    error_msg = "Error processing label $label: $e"
                    push!(log_buffer, error_msg)
                    @warn error_msg
                    @warn sprint(showerror, e, catch_backtrace())
                    lock(results_lock) do
                        push!(skipped_labels, label)
                    end
                    return (label, nothing, log_buffer)
                end
            end
        end

        for task in tasks
            result = fetch(task)
            if !isnothing(result)
                label, features_dict, logs = result

                if p.verbose
                    for line in logs
                        println(line)
                    end
                end

                if !isnothing(features_dict)
                    results[label] = features_dict
                end
            end
        end

        if p.verbose
            println("\n======================")
            println("Multi-label processing completed")
            println("Labels requested: ", p.labels)
            println("Labels successfully processed: ", sort(collect(keys(results))))
            if !isempty(skipped_labels)
                println("Labels skipped (not found or error): ", sort(skipped_labels))
            end
            println("======================")
        end

        return results
    end

    mask_to_use, voxel_count = extract_and_check_mask(p.mask, p.labels)
    if voxel_count == 0
        error("Label $(p.labels) not found in mask (no voxels with this value)")
    end

    if p.verbose
        println("Label $(p.labels) contains $voxel_count voxels")
    end

    img_to_use, mask_to_use = bounding_box(p.img, mask_to_use, p.verbose)

    total_start_time       = time()
    total_time_accumulated = 0.0

    if p.verbose
        println("Active threads: ", Threads.nthreads())
        println("Extracting radiomic features...")
        println("Processing LABEL: $(p.labels)")
        if compute_all
            println("Computing ALL features")
        else
            println("Computing selected features: ", join(string.(p.features), ", "))
        end
        if !isnothing(p.n_bins)
            println("Using n_bins = $(p.n_bins)")
        elseif !isnothing(p.bin_width)
            println("Using bin_width = $(p.bin_width)")
        else
            println("Using default bin width: 25.0")
        end
        if p.keep_largest_only
            println("keep_largest_only = true (will be applied to selected label)")
        end
    end

    radiomic_features, time_acc = _compute_radiomics_impl(
        img_to_use, mask_to_use, p.spacing, voxel_count;
        n_bins            = p.n_bins,
        bin_width         = p.bin_width,
        weighting_norm    = p.weighting_norm,
        verbose           = p.verbose,
        keep_largest_only = p.keep_largest_only,
        features_std      = p.features_std,
        compute_all       = compute_all,
        features          = p.features,
        get_raw_matrices  = p.get_raw_matrices
    )

    total_time_accumulated += time_acc
    total_time_real         = time() - total_start_time

    if p.verbose
        println("\n======================")
        println("Measured time of single function'sum (sum of @timed): $(total_time_accumulated) sec")
        println("Real time (end-to-end): $(total_time_real) sec")
        println("Overhead: $(total_time_real - total_time_accumulated) sec")
        diagnosis_features = get_diagnosis_features(
            p.bin_width, p.spacing, total_time_real,
            p.weighting_norm, p.n_bins, p.keep_largest_only,
            p.img, p.features_std, img_to_use, p.mask, mask_to_use
        )
        merge!(radiomic_features, diagnosis_features)
        print_features_diagnosis("Diagnosis Features", diagnosis_features)
        println("---------------------")
        println("Total features extracted: $(length(radiomic_features))")
        println("Features extraction completed for LABEL $(p.labels)")
        println("======================")
    end

    radiomic_features["label_id"] = p.labels

    return radiomic_features
end

"""
    _compute_radiomics_impl(img, mask, voxel_spacing; 
                           n_bins, bin_width,
                           weighting_norm, verbose, 
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
    - `keep_largest_only`: Keep only largest connected component
    - `compute_all`: Compute all features or only selected ones
    - `features`: Vector of feature symbols to compute
    - `log_buffer`: Optional buffer for collecting log messages (for multi-label parallel processing)
    
    # Returns:
    - Tuple of (radiomic_features::Dict, total_time_accumulated::Float64)
"""
function _compute_radiomics_impl(img::Array{Float64}, mask::BitArray, voxel_spacing::Vector{Float64}, voxel_count::Int;
    n_bins::Union{Nothing,Int}=nothing,
    bin_width::Union{Nothing,Float64}=nothing,
    weighting_norm::Union{Nothing,String}=nothing,
    verbose::Bool=false,
    keep_largest_only::Bool=true,
    compute_all::Bool=true,
    features_std::Bool=false,
    features::Vector{Symbol}=Symbol[],
    get_raw_matrices::Bool=false,
    log_buffer::Union{Nothing,Vector{String}}=nothing)::Tuple{Dict{String,Any}, Float64}
    
    radiomic_features = Dict{String,Any}()
    total_time_accumulated = 0.0
    
    # Helper function to print or buffer log messages
    function log_println(msg::String)
        if isnothing(log_buffer)
            println(msg)
        else
            push!(log_buffer, msg)
        end
    end

    # Validate binning parameters
    if isnothing(n_bins) && !isnothing(bin_width) && voxel_count > 0
        validate_binning_parameters(img, mask, bin_width)
    end

    #Initialize task with nothing
    t_glcm_features = nothing
    t_first_order_features = nothing
    t_glszm_features = nothing
    t_ngtdm_features = nothing
    t_glrlm_features = nothing
    t_gldm_features = nothing
    t_shape3d_features = nothing
    t_shape2d_features = nothing

    # GLCM features
    if compute_all || :glcm in features
        t_glcm_features = Threads.@spawn begin
            result = @timed get_glcm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                weighting_norm=weighting_norm,
                features_std=features_std,
                get_raw_matrices=get_raw_matrices,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end
        
    # First order features
    if compute_all || :first_order in features
        t_first_order_features = Threads.@spawn begin
            result = @timed get_first_order_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end

    # GLSZM features
    if compute_all || :glszm in features
        t_glszm_features = Threads.@spawn begin
            result = @timed get_glszm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                get_raw_matrices=get_raw_matrices,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end

    # NGTDM features
    if compute_all || :ngtdm in features
        t_ngtdm_features = Threads.@spawn begin
            result = @timed get_ngtdm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                get_raw_matrices=get_raw_matrices,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end

    # GLRLM features
    if compute_all || :glrlm in features
        t_glrlm_features = Threads.@spawn begin
            result = @timed get_glrlm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                features_std=features_std,
                weighting_norm=weighting_norm,
                get_raw_matrices=get_raw_matrices,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end

    # GLDM features
    if compute_all || :gldm in features
        t_gldm_features = Threads.@spawn begin
            result = @timed get_gldm_features(
                img, mask, voxel_spacing;
                n_bins=n_bins,
                bin_width=bin_width,
                get_raw_matrices=get_raw_matrices,
                verbose=verbose
            )
            (result.value, result.time)
        end
    end

    # Control dimension of the mask 
    if ndims(mask) == 3
        # 3D shape features
        if compute_all || :shape3d in features
            t_shape3d_features = Threads.@spawn begin
                result = @timed get_shape3d_features(
                    mask, voxel_spacing;
                    verbose=verbose,
                    keep_largest_only=keep_largest_only
                )
                (result.value, result.time)
            end
        end
    end

    # Launch parallel threads for 2D features
    if ndims(mask) == 2
        if compute_all || :shape2d in features
            t_shape2d_features = Threads.@spawn begin
                result = @timed get_shape2d_features(
                    mask, voxel_spacing;
                    verbose=verbose,
                    keep_largest_only=keep_largest_only
                )
                (result.value, result.time)
            end
        end
    end

    # First Order features
    if !isnothing(t_first_order_features)
        first_order_dict, first_order_time = fetch(t_first_order_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, first_order_dict)
        total_time_accumulated += first_order_time
        if verbose
            log_println("First order: $(first_order_time) sec")
            print_features("First Order Features", first_order_dict; log_buffer=log_buffer)
        end
    end

    # GLCM features
    if !isnothing(t_glcm_features)
        glcm_dict, glcm_time = fetch(t_glcm_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, glcm_dict)
        total_time_accumulated += glcm_time
        if verbose
            log_println("GLCM: $(glcm_time) sec")
            if !get_raw_matrices
                print_features("GLCM Features", glcm_dict; log_buffer=log_buffer)
            end
        end
    end
        
    # GLSZM features
    if !isnothing(t_glszm_features)
        glszm_dict, glszm_time = fetch(t_glszm_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, glszm_dict)
        total_time_accumulated += glszm_time
        if verbose
            log_println("GLSZM: $(glszm_time) sec")
            if !get_raw_matrices
                print_features("GLSZM Features", glszm_dict; log_buffer=log_buffer)
            end
        end
    end

    # NGTDM features
    if !isnothing(t_ngtdm_features)
        ngtdm_dict, ngtdm_time = fetch(t_ngtdm_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, ngtdm_dict)
        total_time_accumulated += ngtdm_time
        if verbose
            log_println("NGTDM: $(ngtdm_time) sec")
            if !get_raw_matrices
                print_features("NGTDM Features", ngtdm_dict; log_buffer=log_buffer)
            end
        end
    end

    # GLRLM features
    if !isnothing(t_glrlm_features)
        glrlm_dict, glrlm_time = fetch(t_glrlm_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, glrlm_dict)
        total_time_accumulated += glrlm_time
        if verbose
            log_println("GLRLM: $(glrlm_time) sec")
            if !get_raw_matrices
                print_features("GLRLM Features", glrlm_dict; log_buffer=log_buffer)
            end
        end
    end

    # GLDM features
    if !isnothing(t_gldm_features)
        gldm_dict, gldm_time = fetch(t_gldm_features)::Tuple{Dict{String,Any}, Float64}
        merge!(radiomic_features, gldm_dict)
        total_time_accumulated += gldm_time
        if verbose
            log_println("GLDM: $(gldm_time) sec")
            if !get_raw_matrices
                print_features("GLDM Features", gldm_dict; log_buffer=log_buffer)
            end
        end
    end

    # 3D shape features
    if ndims(mask) == 3
        if !isnothing(t_shape3d_features)
            shape3d_dict, shape3d_time = fetch(t_shape3d_features)::Tuple{Dict{String,Any}, Float64}
            merge!(radiomic_features, shape3d_dict)
            total_time_accumulated += shape3d_time
            if verbose
                log_println("3D shape: $(shape3d_time) sec")
                print_features("3D Shape Features", shape3d_dict; log_buffer=log_buffer)
            end
        end
    end

    # 2D shape features
    if ndims(mask) == 2
        if !isnothing(t_shape2d_features)
            shape2d_dict, shape2d_time = fetch(t_shape2d_features)::Tuple{Dict{String,Any}, Float64}
            merge!(radiomic_features, shape2d_dict)
            total_time_accumulated += shape2d_time
            if verbose
                log_println("2D shape: $(shape2d_time) sec")
                print_features("2D Shape Features", shape2d_dict; log_buffer=log_buffer)
            end
        end
    end
    
    return (radiomic_features, total_time_accumulated)
end

"""
    Wrapper function to be exposed in the C shared library
"""
# Global buffer to avoid the garbage collector (for shared library)
const LAST_JSON_RESULT = Ref{String}("")

Base.@ccallable function c_extract_radiomic_features(
    img_ptr::Ptr{Float64},
    img_size_x::Int64, img_size_y::Int64, img_size_z::Int64,
    mask_ptr::Ptr{Float64},
    spacing_x::Float64, spacing_y::Float64, spacing_z::Float64,
    binWidth::Float64)::Cstring
    global LAST_JSON_RESULT
    try
        # Prepare inputs for the main function
        dims = (Int(img_size_x), Int(img_size_y), Int(img_size_z))
        spacing = [spacing_x, spacing_y, spacing_z]

        img = unsafe_wrap(Array, img_ptr, dims)
        mask = unsafe_wrap(Array, mask_ptr, dims)

        # Call the main function
        c_features_dict = extract_radiomic_features(
            img, mask, spacing;
            bin_width = binWidth,
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

# @setup_workload for precompile and warm up the code.
# - Precompilation: executed automatically at the first `using Radiomics`.
# - Reduces the TTFX (Time To First eXecution) for the package users.
@setup_workload begin
    # Small synthetic data for precompilation warmup
    img_small  = Float64.(reshape(1:1000, 10, 10, 10))
    mask_small = zeros(Float64, 10, 10, 10)
    mask_small[3:7, 3:7, 3:7] .= 1.0

    # Small 2D synthetic data for precompilation warmup
    img_small_2d  = Float64.(reshape(1:100, 10, 10))
    mask_small_2d = zeros(Float64, 10, 10)
    mask_small_2d[3:7, 3:7] .= 1.0

    # 2D mask multi-label
    img_small_2d_multi = Float64.(reshape(1:100, 10, 10))
    mask_small_2d_multi = zeros(Float64, 10, 10)
    mask_small_2d_multi[3:7, 3:7] .= 1.0
    mask_small_2d_multi[6:8, 6:8] .= 2.0
    
    # Multi-label mask
    mask_multi = zeros(Float64, 10, 10, 10)
    mask_multi[2:4, 2:4, 2:4] .= 1.0
    mask_multi[6:8, 6:8, 6:8] .= 2.0

    spacing = [1.0, 1.0, 1.0]

    @compile_workload begin        
        # 2D
        extract_radiomic_features(
            img_small_2d, mask_small_2d, spacing;
            keep_largest_only = true,
            verbose           = false
        )

        # 2D mask multi-label
        extract_radiomic_features(
            img_small_2d_multi, mask_small_2d_multi, spacing;
            keep_largest_only = false,
            verbose           = false
        )

        #2D label
        extract_radiomic_features(
            img_small_2d_multi, mask_small_2d_multi, spacing;
            keep_largest_only = false,
            labels            = [1,2],
            verbose           = false
        )
        
        # --- Default bin_width ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            keep_largest_only = false,
            verbose           = false
        )

        # --- n_bins ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            n_bins            = 32,
            keep_largest_only = false,
            verbose           = false
        )

        # --- bin_width explicit ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            bin_width         = 25.0,
            keep_largest_only = false,
            verbose           = false
        )

        # --- Weighting norms ---
        for wn in ["euclidean", "infinity", "manhattan", "no_weighting"]
            extract_radiomic_features(
                img_small, mask_small, spacing;
                weighting_norm    = wn,
                keep_largest_only = false,
                verbose           = false
            )
        end

        # --- Selective features ---
        for feat in [
            [:glcm],
            [:first_order],
            [:shape3d],
            [:glszm],
            [:ngtdm],
            [:glrlm],
            [:gldm],
            [:glcm, :first_order],
            [:glcm, :glszm, :glrlm, :gldm, :ngtdm],
        ]
            extract_radiomic_features(
                img_small, mask_small, spacing;
                features          = feat,
                keep_largest_only = false,
                verbose           = false
            )
        end

        # --- keep_largest_only = true ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            keep_largest_only = true,
            verbose           = false
        )

        # --- features_std = true ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            features          = [:glrlm],
            features_std      = true,
            keep_largest_only = false,
            verbose           = false
        )

        # --- Multi-label ---
        extract_radiomic_features(
            img_small, mask_multi, spacing;
            labels            = [1, 2],
            keep_largest_only = false,
            verbose           = false
        )

        # --- Single explicit label ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            labels            = 1,
            keep_largest_only = false,
            verbose           = false
        )

        # --- get_raw_matrices ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            get_raw_matrices  = true,
            keep_largest_only = false,
            verbose           = false
        )

        # --- 2D slice extraction ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            slices_2d         = [(1, 5)],
            keep_largest_only = true,
            verbose           = false
        )

        # --- Multiple slices ---
        extract_radiomic_features(
            img_small, mask_small, spacing;
            slices_2d         = [(1, 5), (2, 5), (3, 5)],
            keep_largest_only = false,
            verbose           = false
        )
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
                                        
    # Compute with keep_largest_only personalized (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose = true, keep_largest_only=false);

    # Compute with weighting_norm personalized (uses default label=1)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose = true, weighting_norm="euclidean");
    
    # Extract features for a specific label (e.g., label 2)
    features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; labels=2, verbose=true);
    
    # Extract features for multiple labels in parallel
    results = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; 
                                        labels=[1, 2, 30], 
                                        verbose=true, 
                                        keep_largest_only=true);
    # Access features for label 1: results[1]
    # Access features for label 2: results[2]
    # Access features for label 30: results[30]
    
    # Note: If no label is specified, the function defaults to label=1
"""

end