"""
    _pad_index(i::Int, N::Int, padding::String) -> Int

    Map an out-of-range 1-based index into 1:N according to the padding mode (0 = zero fill).
    Padding modes (IBSI 2 reference manual, sec. 2.2): "zero", "nearest", "periodic", "mirror".
"""
function _pad_index(i::Int, N::Int, padding::String)::Int
    1 <= i <= N && return i
    padding == "zero" && return 0
    padding == "nearest" && return clamp(i, 1, N)
    if padding == "periodic"
        return mod1(i, N)
    end
    # mirror (IBSI): reflect about the image edge with the edge sample repeated (d c b a | a b c d)
    m = mod(i - 1, 2N)
    return m < N ? m + 1 : 2N - m
end

"""
    _conv1d_axis(data, axis, kernel, padding) -> Array{Float64}

    Filter `data` along `axis` with an odd-length kernel, handling the boundary via `padding`.
    Computes out[i] = Σ_t kernel[t] * data[i + t - r - 1] (correlation, kernel centred at r + 1).
    For a symmetric kernel this equals convolution; for a non-symmetric kernel (e.g. wavelets)
    pass `reverse(kernel)` to obtain a true convolution.
"""
function _conv1d_axis(data::Array{Float64}, axis::Int, kernel::Vector{Float64}, padding::String)::Array{Float64}
    r = length(kernel) ÷ 2
    N = size(data, axis)
    out = zeros(Float64, size(data))
    # reshape to (before, N, after) so the filtered axis is the middle one
    nb = prod(size(data)[1:axis-1])
    na = prod(size(data)[axis+1:end])
    d3 = reshape(data, nb, N, na)
    o3 = reshape(out, nb, N, na)
    idx = [_pad_index(i + o, N, padding) for i in 1:N, o in -r:r]  # N × M, 0 = zero fill
    Threads.@threads for c in 1:na
        @inbounds for i in 1:N, t in eachindex(kernel)
            src = idx[i, t]
            src == 0 && continue
            w = kernel[t]
            for b in 1:nb
                o3[b, i, c] += w * d3[b, src, c]
            end
        end
    end
    return out
end

"""
    _right_angle_rotations(Nd::Int) -> Vector{Tuple{Vector{Int},Vector{Bool}}}

    Right-angle rotations of a 2D/3D array (IBSI 2 reference manual, sec. 3.3 and appendix A).
    A rotation is an axis permutation followed by flips; only proper rotations (determinant +1)
    are kept: 4 in 2D, 24 in 3D. Rotating the image, filtering and rotating back is equivalent to
    filtering with the rotated kernel, provided the kernels have odd length (see _atrous_kernel).

    Nd::Int - number of dimensions of the array

    Returns:
    Vector{Tuple{Vector{Int},Vector{Bool}}} - list of right-angle rotations

    Example:
    _right_angle_rotations(2) returns [([1,2], [false,false]), ([2,1], [true,false]), ([1,2], [true,true]), ([2,1], [false,true])]
""" 
function _right_angle_rotations(Nd::Int)::Vector{Tuple{Vector{Int},Vector{Bool}}}
    Nd ∈ (2, 3) || error("rotations are defined for 2D or 3D images, got $(Nd)D")
    perms(v) = length(v) <= 1 ? [copy(v)] :
               [vcat(v[i], p) for i in eachindex(v) for p in perms(deleteat!(copy(v), i))]
    # sign of a permutation from the number of inversions
    perm_sign(p) = iseven(count(p[i] > p[j] for i in eachindex(p) for j in i+1:length(p))) ? 1 : -1

    rots = Tuple{Vector{Int},Vector{Bool}}[]
    for p in perms(collect(1:Nd)), f in Iterators.product(fill((false, true), Nd)...)
        flips = collect(f)
        perm_sign(p) * prod(fl ? -1 : 1 for fl in flips) == 1 && push!(rots, (p, flips))
    end
    return rots
end

"""
    _rotate(x::Array{Float64}, perm::Vector{Int}, flips::Vector{Bool}) -> Array{Float64}

    Rotate an array by permuting axes and flipping some of them (IBSI 2 reference manual, sec. 3.3 and appendix A).
    The rotation is performed by permuting the axes of `x` according to `perm` and then flipping the axes specified by `flips`.

    x::Array{Float64} - array to rotate
    perm::Vector{Int} - permutation of the axes
    flips::Vector{Bool} - whether to flip each axis

    Returns:
    Array{Float64} - rotated array

    Example:
    _rotate([1 2; 3 4], [2,1], [false,true]) returns [2 1; 4 3]
"""
function _rotate(x::Array{Float64}, perm::Vector{Int}, flips::Vector{Bool})::Array{Float64}
    y = permutedims(x, perm)
    for (a, fl) in enumerate(flips)
        fl && (y = reverse(y; dims=a))
    end
    return y
end

"""
    _rotate_back(y::Array{Float64}, perm::Vector{Int}, flips::Vector{Bool}) -> Array{Float64}

    Rotate an array back by permuting axes and flipping some of them (IBSI 2 reference manual, sec. 3.3 and appendix A).
    The rotation is performed by permuting the axes of `y` according to `perm` and then flipping the axes specified by `flips`.

    y::Array{Float64} - array to rotate
    perm::Vector{Int} - permutation of the axes
    flips::Vector{Bool} - whether to flip each axis

    Returns:
    Array{Float64} - rotated array

    Example:
    _rotate_back([2 1; 4 3], [2,1], [false,true]) returns [1 2; 3 4]
"""
function _rotate_back(y::Array{Float64}, perm::Vector{Int}, flips::Vector{Bool})::Array{Float64}
    for (a, fl) in enumerate(flips)
        fl && (y = reverse(y; dims=a))
    end
    return permutedims(y, invperm(perm))
end

"""
    _rotation_pooled(f, data, pooling) -> Dict{String,Array{Float64}}

    Pseudo rotation-invariant filtering (IBSI 2, sec. 3.3): `f` (image → Dict of response maps) is
    applied to every right-angle rotation of `data`; each response map is rotated back and the maps
    with the same key are pooled over the rotations with `pooling` ("average", "max" or "min").
"""
function _rotation_pooled(f::Function, data::Array{Float64}, pooling::String)::Dict{String,Array{Float64}}
    pooling in ("average", "max", "min") ||
        error("Invalid pooling \"$pooling\". Valid options: average, max, min")
    rots = _right_angle_rotations(ndims(data))
    pooled = Dict{String,Array{Float64}}()
    for (perm, flips) in rots
        for (key, map) in f(_rotate(data, perm, flips))
            r = _rotate_back(map, perm, flips)
            if !haskey(pooled, key)
                pooled[key] = r
            elseif pooling == "average"
                pooled[key] .+= r
            elseif pooling == "max"
                pooled[key] .= max.(pooled[key], r)
            else
                pooled[key] .= min.(pooled[key], r)
            end
        end
    end
    if pooling == "average"
        for key in keys(pooled)
            pooled[key] ./= length(rots)
        end
    end
    return pooled
end


# ---------------------------------------------------------------------------
# Undecimated (à trous) separable wavelet decomposition (IBSI 2 reference manual, sec. 4.6).
# Generic for every separable wavelet: Haar, Daubechies, Coiflet only differ in the level-1 kernels.
# ---------------------------------------------------------------------------


"""
    _atrous_kernel(base, level) -> Vector{Float64}

    Kernel of decomposition `level` (≥ 1) built from the level-1 kernel `base` (IBSI 2, sec. 4.6.2):
    each coefficient is followed by 2^(level-1) - 1 zeros (pywavelets / MATLAB convention), then a 0
    is appended if the length is even, so the kernel has odd length (IBSI 2, sec. 3.3 and appendix A).

    e.g. Haar high-pass: level 1 → [-1/√2, 1/√2, 0], level 2 → [-1/√2, 0, 1/√2, 0, 0].
"""
function _atrous_kernel(base::Vector{Float64}, level::Int)::Vector{Float64}
    level >= 1 || error("level must be ≥ 1")
    step = 2^(level - 1)
    k = zeros(Float64, length(base) * step)
    k[1:step:end] .= base
    iseven(length(k)) && push!(k, 0.0)
    return k
end

"""
    _separable_wavelet_approx(data, lo, nlevels, padding) -> Array{Float64}

    Approximation after `nlevels` decomposition levels: the low-pass kernel of levels 1, …, `nlevels`
    is applied along every axis. `nlevels = 0` returns a copy of `data`.
"""
function _separable_wavelet_approx(data::Array{Float64}, lo::Vector{Float64},
    nlevels::Int, padding::String)::Array{Float64}
    out = copy(data)
    for j in 1:nlevels
        # reverse: _conv1d_axis computes a correlation, the wavelet transform is a convolution
        kL = reverse(_atrous_kernel(lo, j))
        for d in 1:ndims(out)
            out = _conv1d_axis(out, d, kL, padding)
        end
    end
    return out
end

"""
    _separable_wavelet_detail(approx, lo, hi, level, subband, padding) -> Array{Float64}

    Response map of `subband` (e.g. "LLH") at decomposition `level`, computed from the approximation
    of the previous level (`_separable_wavelet_approx(data, lo, level - 1, padding)`): along axis d
    the low-pass (`'L'`) or high-pass (`'H'`) kernel of `level` is applied, according to the d-th
    character of `subband`.
"""
function _separable_wavelet_detail(approx::Array{Float64}, lo::Vector{Float64}, hi::Vector{Float64},
    level::Int, subband::String, padding::String)::Array{Float64}
    Nd = ndims(approx)
    length(subband) == Nd || error("subband \"$subband\" must have $Nd characters for a $(Nd)D image")
    out = approx
    for (d, c) in enumerate(subband)
        base = c == 'L' ? lo : c == 'H' ? hi : error("Invalid subband character '$c' in \"$subband\"")
        out = _conv1d_axis(out, d, reverse(_atrous_kernel(base, level)), padding)
    end
    return out
end