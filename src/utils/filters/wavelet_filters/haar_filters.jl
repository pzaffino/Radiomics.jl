const _HAAR_INV_SQRT2 = 1.0 / sqrt(2.0)
const _HAAR_LO = [_HAAR_INV_SQRT2, _HAAR_INV_SQRT2]    # scaling function (low-pass)
const _HAAR_HI = [-_HAAR_INV_SQRT2, _HAAR_INV_SQRT2]   # wavelet (high-pass)

"""
    haar_wavelet_filter(img, spacing; level=1, start_level=0, subbands="all",
                        padding="mirror", rotation_invariant=true, pooling="average")

    IBSI-compliant undecimated (stationary) Haar wavelet transform (IBSI 2 reference manual, sec. 4.6).

    Level-1 kernels g_L = [1/√2, 1/√2] and g_H = [-1/√2, 1/√2]; the kernels of level j are obtained with
    the à trous algorithm and extended with a trailing 0 to odd length (see `_atrous_kernel`).
    The subband of level j is computed from the approximation of level j-1: along axis d the low-pass
    ('L') or high-pass ('H') kernel is applied according to the d-th character of the subband name
    (first character = dim 1, second = dim 2, third = dim 3).

    With `rotation_invariant=true` the transform is applied to all right-angle rotations of the image
    (24 in 3D, 4 in 2D) and the response maps are pooled (IBSI 2, sec. 3.3 and appendix A).
    After pooling, subbands with the same number of 'H' are identical (e.g. LLH = LHL = HLL):
    - `subbands="all"` returns only the distinct ones (3D: LLL, LLH, LHH, HHH; 2D: LL, LH, HH) and logs an `@info`;
    - explicitly requested subbands are all returned, with a `@warn` listing their equivalents.

    Returns a `Dict{String, Array{Float64}}` with keys:
    - `"wavelet-<subband>"` for the first computed level, `"wavelet<k>-<subband>"` for the k-th (k ≥ 2);
    - the all-low-pass subband (`LLL` / `LL`) only for the last computed level.

    IBSI 2 tests 7.a.1 / 7.a.2: `level=2`, `padding="mirror"`, `rotation_invariant=true`, `pooling="average"`.

    # Arguments
    - `img`: 2D or 3D image (any real-valued array; converted to Float64 internally).
    - `spacing::AbstractVector{<:Real}`: voxel spacing in mm. Not used by the Haar transform, which works
    in voxel units; accepted to match the common filter signature.
    - `level::Int=1`: number of decomposition levels returned.
    - `start_level::Int=0`: number of approximation levels computed (and not returned) before the first returned level.
    - `subbands::Union{String,Vector{String}}="all"`: `"all"` or a vector of subband names (e.g. `["LLL", "HHH"]`).
    - `padding::String="mirror"`: boundary condition: `"mirror"`, `"zero"`, `"nearest"` or `"periodic"`.
    - `rotation_invariant::Bool=true`: pool the response maps over right-angle rotations.
    - `pooling::String="average"`: pooling over rotations: `"average"`, `"max"` or `"min"`.
"""
function haar_wavelet_filter(img::AbstractArray{<:Real}, spacing::AbstractVector{<:Real};
    level::Int=1,
    start_level::Int=0,
    subbands::Union{String,Vector{String}}="all",
    padding::String="mirror",
    rotation_invariant::Bool=true,
    pooling::String="average")::Dict{String,Array{Float64}}

    Nd = ndims(img)
    Nd ∈ (2, 3) || error("img must be 2D or 3D, got $(Nd)D")
    level >= 1 || error("level must be ≥ 1")
    start_level >= 0 || error("start_level must be ≥ 0")
    padding in ("zero", "mirror", "periodic", "nearest") ||
        error("Invalid padding \"$padding\". Valid options: zero, mirror, periodic, nearest")
    pooling in ("average", "max", "min") ||
        error("Invalid pooling \"$pooling\". Valid options: average, max, min")

    valid = sort(vec([join(c) for c in Iterators.product(fill(('L', 'H'), Nd)...)]))
    all_low = "L"^Nd

    requested = if subbands == "all"
        if rotation_invariant
            distinct = ["L"^(Nd - n) * "H"^n for n in 0:Nd]
            @info "rotation_invariant=true: subbands with the same number of H are identical after rotation pooling " *
                  "(e.g. $(join([v for v in valid if count(==('H'), v) == 1], " = "))); returning only $(join(distinct, ", "))"
            distinct
        else
            valid
        end
    else
        subbands isa Vector{String} || error("subbands must be \"all\" or a Vector{String}, got \"$subbands\"")
        isempty(subbands) && error("subbands must contain at least one entry")
        for sb in subbands
            sb in valid || error("Invalid subband \"$sb\". Valid options for $(Nd)D: $valid")
        end
        if rotation_invariant
            for sb in unique(subbands)
                n = count(==('H'), sb)
                0 < n < Nd || continue   # all-L and all-H subbands have no equivalents
                equivalent = [v for v in valid if count(==('H'), v) == n]
                @warn "rotation_invariant=true: $(join(equivalent, " = ")) after rotation pooling (requested $sb)"
            end
        end
        unique(subbands)
    end

    detail_subbands = filter(!=(all_low), requested)
    include_approx = all_low in requested

    decompose(x) = _haar_decomposition(x, level, start_level, detail_subbands, include_approx, padding)

    data = collect(Float64, img)
    return rotation_invariant ? _rotation_pooled(decompose, data, pooling) : decompose(data)
end

# Haar decomposition of one (possibly rotated) image: details of every returned level and,
# if requested, the approximation of the last level, keyed as in haar_wavelet_filter.

"""
    _haar_decomposition(x::Array{Float64}, level::Int, start_level::Int,
    detail_subbands::Vector{String}, include_approx::Bool, padding::String)::Dict{String,Array{Float64}}

    Haar decomposition of one (possibly rotated) image: details of every returned level and,
    if requested, the approximation of the last level, keyed as in haar_wavelet_filter.
"""
function _haar_decomposition(x::Array{Float64}, level::Int, start_level::Int,
    detail_subbands::Vector{String}, include_approx::Bool, padding::String)::Dict{String,Array{Float64}}
    all_low = "L"^ndims(x)
    a = _separable_wavelet_approx(x, _HAAR_LO, start_level, padding)
    out = Dict{String,Array{Float64}}()
    for idx in 1:level
        j = start_level + idx                       # absolute decomposition level
        prefix = idx == 1 ? "wavelet" : "wavelet$idx"
        for sb in detail_subbands
            out["$prefix-$sb"] = _separable_wavelet_detail(a, _HAAR_LO, _HAAR_HI, j, sb, padding)
        end
        # approximation of level j (needed for the next level, or returned at the last one)
        (idx < level || include_approx) &&
            (a = _separable_wavelet_detail(a, _HAAR_LO, _HAAR_HI, j, all_low, padding))
    end
    if include_approx
        prefix = level == 1 ? "wavelet" : "wavelet$level"
        out["$prefix-$all_low"] = a
    end
    return out
end