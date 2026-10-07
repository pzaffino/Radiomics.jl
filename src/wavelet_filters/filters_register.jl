# Map of family_name => filter function.
# Each filter must have the same signature as haar_wavelet_filter:
#   (img::AbstractArray{<:Real}; level::Int, start_level::Int, subbands::Union{String,Vector{String}}) -> Dict{String,Array{Float64}}
const WAVELET_FILTERS = Dict{String,Function}(
    "haar" => haar_wavelet_filter,
    # "db2"   => db2_wavelet_filter,    # to be implemented
    # "sym4"  => sym4_wavelet_filter,   # to be implemented
    # "coif1" => coif1_wavelet_filter,  # to be implemented
)

"""
    get_wavelet_filter(wavelet_type::String) -> Filter

    Retrieve the filter coefficients associated with the specified wavelet type.

    # Arguments
    - `wavelet_type::String`: The name of the requested wavelet (e.g., "db2", "haar").

    # Throws
    - `ErrorException`: If the provided string does not match any supported wavelet type.
"""
function get_wavelet_filter(wavelet_type::String)
    haskey(WAVELET_FILTERS, wavelet_type) || error(
        "Unsupported wavelet_type \"$wavelet_type\". Available: $(sort(collect(keys(WAVELET_FILTERS))))"
    )
    return WAVELET_FILTERS[wavelet_type]
end