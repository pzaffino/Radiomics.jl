"""
    log_filter(img, spacing; sigma, truncate=4.0, padding="zero", dims=3)

IBSI-compliant Laplacian of Gaussian filter (IBSI 2 reference manual, section 4.3).

The kernel is the radial second derivative of a `D`-dimensional Gaussian (Eq. 4.2),

    g[k] = -1/σ² · (1/(√(2π)σ))^D · (D - ‖k‖²/σ²) · exp(-‖k‖²/(2σ²)),

with `σ = σ*/s` in voxel units (`σ*` in mm, `s` the voxel spacing, per axis). The kernel is truncated
to `M = 1 + 2⌊d·σ + 0.5⌋` samples per axis (`d = truncate`) and is **not** scale-normalised.
Because the Laplacian is a sum of separable terms, the filter is computed with 1D convolutions; the
result is identical to convolving with the full truncated `D`-dimensional kernel.

# Arguments
- `img`: 2D or 3D real array.
- `spacing`: voxel spacing, one entry per image dimension (mm).
- `sigma`: scale `σ*` in mm. A number returns one image; a vector returns one image per entry.
- `truncate`: kernel truncation `d` (default 4).
- `padding`: `"zero"` (default), `"mirror"` (edge sample repeated, as IBSI), `"periodic"` or `"nearest"`.
- `dims`: `3` (default) filters in 3D; `2` filters slice-wise in the plane of the first two dimensions.
  A 2D image only supports `dims=2`.

Returns a `Dict{String,Array{Float64}}` keyed pyradiomics-style, `"log-sigma-<σ>-mm-3D"` (or `-2D`),
where `.` in `σ` is replaced by `-` (e.g. `"log-sigma-3-0-mm-3D"`).
"""
function log_filter(img::AbstractArray{<:Real}, spacing::AbstractVector{<:Real};
    sigma::Union{Real,AbstractVector{<:Real}},
    truncate::Real=4.0,
    padding::String="zero",
    dims::Int=3)::Dict{String,Array{Float64}}

    Nd = ndims(img)
    Nd ∈ (2, 3) || error("img must be 2D or 3D, got $(Nd)D")
    length(spacing) == Nd || error("spacing must have $(Nd) elements, got $(length(spacing))")
    dims ∈ (2, 3) || error("dims must be 2 or 3, got $dims")
    dims == 3 && Nd == 2 && error("A 2D image requires dims=2")
    truncate > 0 || error("truncate must be > 0")
    padding in ("zero", "mirror", "periodic", "nearest") ||
        error("Invalid padding \"$padding\". Valid options: zero, mirror, periodic, nearest")
    sigmas = sigma isa Real ? [Float64(sigma)] : Float64.(collect(sigma))
    isempty(sigmas) && error("sigma must contain at least one value")
    all(>(0), sigmas) || error("sigma values must be > 0")
    all(>(0), spacing) || error("spacing values must be > 0")

    data = collect(Float64, img)
    sp = Float64.(collect(spacing))
    result = Dict{String,Array{Float64}}()
    for sg in sigmas
        key = "log-sigma-$(replace(string(sg), "." => "-"))-mm-$(dims)D"
        result[key] = _log_response(data, sp, sg, Float64(truncate), padding, dims)
    end
    return result
end

# Sum over axes i of  (∂²G_i) ⊗ (⊗_{j≠i} G_j), each term evaluated with separable 1D convolutions.
function _log_response(data::Array{Float64}, sp::Vector{Float64}, sigma_mm::Float64,
    truncate::Float64, padding::String, dims::Int)::Array{Float64}
    axes_f = 1:dims                       # axes that are filtered
    sig_vox = [sigma_mm / sp[a] for a in axes_f]
    g0 = [_gauss_kernel(sig_vox[a], truncate, 0) for a in axes_f]
    g2 = [_gauss_kernel(sig_vox[a], truncate, 2) for a in axes_f]

    out = zeros(Float64, size(data))
    for i in axes_f
        term = data
        for a in axes_f
            term = _conv1d_axis(term, a, a == i ? g2[a] : g0[a], padding)
        end
        out .+= term
    end
    return out
end

# Sampled 1D Gaussian (order 0) or its second derivative (order 2) in voxel units,
# truncated to M = 1 + 2⌊d σ + 0.5⌋ samples. Sum of order-0 kernel is ≈ 1 (not renormalised).
function _gauss_kernel(sigma::Float64, truncate::Float64, order::Int)::Vector{Float64}
    r = floor(Int, truncate * sigma + 0.5)
    k = collect(-r:r)
    g = @. exp(-k^2 / (2 * sigma^2)) / (sqrt(2π) * sigma)
    order == 0 && return g
    return @. (k^2 / sigma^4 - 1 / sigma^2) * g
end

# Convolve along `axis` with a symmetric, odd-length kernel, handling the boundary via `padding`.
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