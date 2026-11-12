# Map of filter_name => filter function.
# Wavelets and convolutional filters share one register and one signature:
#   (img::AbstractArray{<:Real}, spacing::AbstractVector{<:Real}; kwargs...) -> Dict{String,Array{Float64}}
# - `spacing`: voxel spacing in mm, one value per image dimension (filters that work in voxel units may ignore it).
# - `kwargs`: filter-specific parameters, forwarded from `filter_params` in extract_radiomic_features
#   (e.g. `level`/`subbands` for wavelets, `sigma` for LoG).
# - Return value: one or more response maps, keyed by a name that identifies the map
#   (e.g. "LLH" for a wavelet subband, "log-sigma-1-0-mm-3D" for LoG).
const FILTERS = Dict{String,Function}(
    # Wavelets (wavelet_filters/)
    "haar"  => haar_wavelet_filter,
    # "db2"   => db2_wavelet_filter,    # to be implemented
    # "coif1" => coif1_wavelet_filter,  # to be implemented

    # Convolutional filters (convolutional_filters/)
    "log"   => log_filter,
    "gabor" => gabor_filter,
    # "mean"  => mean_filter,           # to be implemented
    # "laws"  => laws_filter,           # to be implemented
)

"""
    get_filter(filter_type::String) -> Function

    Retrieve the filter function associated with the specified filter type.

    # Arguments
    - `filter_type::String`: The name of the requested filter (e.g., "haar", "log", "gabor").

    # Throws
    - `ErrorException`: If the provided string does not match any supported filter type.
"""
function get_filter(filter_type::String)
    haskey(FILTERS, filter_type) || error(
        "Unsupported filter_type \"$filter_type\". Available: $(sort(collect(keys(FILTERS))))"
    )
    return FILTERS[filter_type]
end