
const _HAAR_INV_SQRT2 = 1.0 / sqrt(2.0)

"""
    haar_wavelet_filter(img; level=1, start_level=0, subbands=nothing)

Applies the Haar Stationary Wavelet Transform (SWT, à-trous algorithm) to `img`,
matching pyradiomics' `getWaveletImage` / `_swt3` implementation exactly.

Uses pywt's Haar filter bank (same as pyradiomics):
  dec_lo = [1/√2, 1/√2]   → y_L[i] = (x[i] + x[i+1]) / √2  (current + next, periodic)
  dec_hi = [-1/√2, 1/√2]  → y_H[i] = (x[i] - x[i+1]) / √2  (current - next, periodic)

where index i+1 wraps to 1 when i == N (periodic extension).

Returns a `Dict{String, Array{Float64}}` whose keys follow pyradiomics naming:
- `"wavelet-LLH"`, `"wavelet-LHL"`, …, `"wavelet-HHH"` — detail subbands, level 1
- `"wavelet-LLL"` — final approximation (level 1)
- For multi-level: `"wavelet2-LLL"`, `"wavelet2-LLH"`, … (prefix = `"wavelet<k>"`)

Key character order matches pyradiomics (axes iterated as `(2,1,0)` in NumPy):
first character = dim 1 / X-axis, second = dim 2 / Y-axis, third = dim 3 / Z-axis.

# Arguments
- `img`: 2D or 3D image (any real-valued array; converted to Float64 internally)
- `level::Int=1`: number of decomposition levels
- `start_level::Int=0`: skip this many levels of approximation before starting
- `subbands::Union{String, Vector{String}}="all"`: controls which subbands are returned.
  `"all"` (default) returns all 8 subbands. A `Vector{String}` returns only the listed
  ones (e.g. `["LLL", "HHH"]`). For 3D images the valid names are all combinations of
  L/H of length 3; for 2D, length 2. Unused branches of the filter tree are skipped,
  so requesting fewer subbands is faster.
"""
function haar_wavelet_filter(img::AbstractArray{<:Real};
    level::Int=1,
    start_level::Int=0,
    subbands::Union{String,Vector{String}}="all")::Dict{String,Array{Float64}}
    Nd = ndims(img)
    Nd ∈ (2, 3) || error("img must be 2D or 3D, got $(Nd)D")
    level >= 1 || error("level must be ≥ 1")
    start_level >= 0 || error("start_level must be ≥ 0")

    # Normalise: "all" → compute everything; a Vector → filter to requested subbands
    requested = if subbands == "all"
        nothing
    else
        valid = Set(join(c) for c in Iterators.product(fill(('L', 'H'), Nd)...))
        isempty(subbands) && error("subbands must contain at least one entry")
        for sb in subbands
            sb in valid || error("Invalid subband \"$sb\". Valid options for $(Nd)D: $(sort(collect(valid)))")
        end
        subbands
    end

    data = collect(Float64, img)
    data, padded_dims = _wrap_pad_odd_dims(data)

    # Advance to start_level by iterating approximations only (no stored subbands)
    for _ in 1:start_level
        data = _haar_swt_approx(data)
    end

    all_detail_subbands = Vector{Dict{String,Array{Float64}}}()
    approx_key = "a"^Nd

    for _ in 1:level
        # Build the set of internal (a/d) keys needed at this level.
        # The all-approx key is always required to advance data to the next level.
        needed = if isnothing(requested)
            nothing
        else
            s = Set{String}(replace(replace(sb, "L" => "a"), "H" => "d") for sb in requested)
            push!(s, approx_key)
            s
        end

        computed = _haar_swt_nd(data; needed=needed)
        data = computed[approx_key]

        named = Dict{String,Array{Float64}}()
        for (k, v) in computed
            k == approx_key && continue
            named[replace(replace(k, "a" => "L"), "d" => "H")] = _crop_padded_dims(v, padded_dims)
        end
        push!(all_detail_subbands, named)
    end

    result = Dict{String,Array{Float64}}()
    n_levels = length(all_detail_subbands)

    for (idx, level_subbands) in enumerate(all_detail_subbands)
        prefix = idx == 1 ? "wavelet" : "wavelet$idx"
        for (k, v) in level_subbands
            result["$prefix-$k"] = v
        end
    end

    # Include the final approximation (LLL) only if requested (or if no filter applied)
    if isnothing(requested) || "L"^Nd in requested
        approx_prefix = n_levels == 1 ? "wavelet" : "wavelet$n_levels"
        result["$approx_prefix-$("L"^Nd)"] = _crop_padded_dims(data, padded_dims)
    end

    return result
end

function _wrap_pad_odd_dims(data::Array{Float64})
    padded_dims = falses(ndims(data))
    result = data
    for dim in 1:ndims(result)
        if size(result, dim) % 2 != 0
            idx = ntuple(i -> i == dim ? (1:1) : Colon(), ndims(result))
            result = cat(result, result[idx...]; dims=dim)
            padded_dims[dim] = true
        end
    end
    return result, padded_dims
end

function _crop_padded_dims(data::Array{Float64}, padded_dims::AbstractVector{Bool})
    idx = ntuple(ndims(data)) do dim
        padded_dims[dim] ? (1:size(data, dim)-1) : Colon()
    end
    return data[idx...]
end

# Apply low-pass along every dimension sequentially to produce only the approximation.
function _haar_swt_approx(data::Array{Float64})::Array{Float64}
    result = data
    for dim in 1:ndims(data)
        result, _ = _haar_swt1d(result, dim)
    end
    return result
end

# Apply 1D Haar SWT along every dimension (dim 1 first = X-axis first, matching
# pyradiomics' axes=(2,1,0) in NumPy ZYX convention where axis 2 = X).
# Returns a Dict whose keys are strings of 'a'/'d' characters, one per dimension.
# When `needed` is given, branches whose key is not a prefix of any needed key are skipped.
function _haar_swt_nd(data::Array{Float64}; needed::Union{Nothing,Set{String}}=nothing)::Dict{String,Array{Float64}}
    Nd = ndims(data)
    subbands = Dict{String,Array{Float64}}("" => data)

    for dim in 1:Nd
        new_subbands = Dict{String,Array{Float64}}()
        for (key, arr) in subbands
            lo, hi = _haar_swt1d(arr, dim)
            key_a = key * "a"
            key_d = key * "d"
            if isnothing(needed) || any(startswith(r, key_a) for r in needed)
                new_subbands[key_a] = lo
            end
            if isnothing(needed) || any(startswith(r, key_d) for r in needed)
                new_subbands[key_d] = hi
            end
        end
        subbands = new_subbands
    end

    return subbands
end

# Apply 1D Haar SWT along dimension `dim` with periodic boundary.
# For dim != 1, permutes that dim to position 1, applies the filter, permutes back.
function _haar_swt1d(data::Array{Float64}, dim::Int)::Tuple{Array{Float64},Array{Float64}}
    if dim == 1
        return _haar_swt1d_along_first(data)
    end
    perm = [dim; [i for i in 1:ndims(data) if i != dim]]
    d = permutedims(data, perm)
    lo_p, hi_p = _haar_swt1d_along_first(d)
    inv_p = invperm(perm)
    return permutedims(lo_p, inv_p), permutedims(hi_p, inv_p)
end

# Haar SWT along dimension 1, matching pywt's exact behavior:
#   dec_lo = [1/√2, 1/√2]:  y_L[i] = (x[i] + x[i+1]) / √2  (current + next)
#   dec_hi = [-1/√2, 1/√2]: y_H[i] = (x[i] - x[i+1]) / √2  (current - next)
# Index i+1 wraps to 1 when i == N (periodic extension).
function _haar_swt1d_along_first(data::Array{Float64})::Tuple{Array{Float64},Array{Float64}}
    out_lo = similar(data)
    out_hi = similar(data)
    N = size(data, 1)
    data_r = reshape(data, N, :)
    out_lo_r = reshape(out_lo, N, :)
    out_hi_r = reshape(out_hi, N, :)
    M = size(data_r, 2)

    @inbounds for j in 1:M
        for i in 1:N
            x_curr = data_r[i, j]
            x_next = data_r[i == N ? 1 : i + 1, j]
            out_lo_r[i, j] = (x_curr + x_next) * _HAAR_INV_SQRT2
            out_hi_r[i, j] = (x_curr - x_next) * _HAAR_INV_SQRT2
        end
    end

    return out_lo, out_hi
end
