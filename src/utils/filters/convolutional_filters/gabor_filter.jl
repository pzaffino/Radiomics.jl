"""
    gabor_filter(img, spacing; sigma, lambda, gamma=1.0, theta=0.0,
                 response="modulus", padding="zero",
                 rotation_invariant=false, delta_theta=π/4, pooling="average",
                 orthogonal_planes=false)

    IBSI-compliant 2D Gabor filter (IBSI 2 reference manual, section 4.5).

    The complex kernel is

        g[k] = exp(-(k̃₁² + γ² k̃₂²) / (2σ²) + j 2π k̃₁ / λ),   (k̃₁, k̃₂) = R_θ k,
        R_θ  = [cos θ  sin θ; sin θ  -cos θ]

    with `σ` and `λ` given in physical units (e.g. mm) and `k` the physical offset from the kernel
    centre, i.e. `σ = σ*/s` and `λ = λ*/s` in voxel units. The kernel is cropped to the image size
    (`M = N` if `N` is odd, `N + 1` if `N` is even, Eq. 4.5) and convolved with the image.

    The filter is applied slice-wise in the plane spanned by the first two array dimensions.
    A 2D input is filtered directly.

    # Arguments
    - `img`: 2D or 3D real array.
    - `spacing`: voxel spacing (same length as `ndims(img)`), in the unit of `sigma` and `lambda`.
    - `sigma`: scale of the Gaussian envelope.
    - `lambda`: wavelength of the oscillation.
    - `gamma`: spatial aspect ratio (ellipticity). Default 1.
    - `theta`: orientation in radians. Default 0.
    - `response`: `"modulus"` (default), `"real"`, `"imaginary"` or `"phase"`.
    - `padding`: `"zero"` (default), `"mirror"` (edge sample repeated, as IBSI), `"periodic"` or `"nearest"`.
    - `rotation_invariant`: if true, pool response maps over orientations
    `θ + i·delta_theta`, `i = 0, …, ⌈2π/delta_theta⌉ - 1`.
    - `pooling`: `"average"` (default), `"max"` or `"min"`.
    - `orthogonal_planes`: if true (3D only), the 2D filter is applied in the three orthogonal
    planes and the three response maps are averaged.

    Returns a one-entry `Dict{String,Array{Float64}}` keyed by the response type
    (e.g. `"modulus"` => response map with the size of `img`).
"""
function gabor_filter(img::AbstractArray{<:Real}, spacing::AbstractVector{<:Real};
    sigma::Real,
    lambda::Real,
    gamma::Real=1.0,
    theta::Real=0.0,
    response::String="modulus",
    padding::String="zero",
    rotation_invariant::Bool=false,
    delta_theta::Real=π / 4,
    pooling::String="average",
    orthogonal_planes::Bool=false)::Dict{String,Array{Float64}}

    Nd = ndims(img)
    Nd ∈ (2, 3) || error("img must be 2D or 3D, got $(Nd)D")
    length(spacing) == Nd || error("spacing must have $(Nd) elements, got $(length(spacing))")
    sigma > 0 || error("sigma must be > 0")
    lambda > 0 || error("lambda must be > 0")
    gamma > 0 || error("gamma must be > 0")
    delta_theta > 0 || error("delta_theta must be > 0")
    response in ("modulus", "real", "imaginary", "phase") ||
        error("Invalid response \"$response\". Valid options: modulus, real, imaginary, phase")
    padding in ("zero", "mirror", "periodic", "nearest") ||
        error("Invalid padding \"$padding\". Valid options: zero, mirror, periodic, nearest")
    pooling in ("average", "max", "min") ||
        error("Invalid pooling \"$pooling\". Valid options: average, max, min")
    orthogonal_planes && Nd == 2 && error("orthogonal_planes requires a 3D image")

    data = collect(Float64, img)
    sp = Float64.(collect(spacing))

    thetas = if rotation_invariant
        n = ceil(Int, 2π / delta_theta - 1e-9)
        [theta + (i - 1) * delta_theta for i in 1:n]
    else
        [Float64(theta)]
    end

    if !orthogonal_planes
        h = _gabor_pooled(data, sp, 1, 2, Float64(sigma), Float64(lambda), Float64(gamma),
            thetas, response, padding, pooling)
        return Dict{String,Array{Float64}}(response => h)
    end

    # Orthogonal planes: (dim1, dim2), (dim1, dim3), (dim2, dim3), averaged.
    acc = zeros(Float64, size(data))
    for (a, b) in ((1, 2), (1, 3), (2, 3))
        acc .+= _gabor_pooled(data, sp, a, b, Float64(sigma), Float64(lambda), Float64(gamma),
            thetas, response, padding, pooling)
    end

    return Dict{String,Array{Float64}}(response => acc ./ 3)
    
end

# Pool response maps over `thetas`, filtering in the plane (dim_a, dim_b).
function _gabor_pooled(data::Array{Float64}, sp::Vector{Float64}, dim_a::Int, dim_b::Int,
    sigma::Float64, lambda::Float64, gamma::Float64, thetas::Vector{Float64},
    response::String, padding::String, pooling::String)::Array{Float64}

    out = nothing
    for th in thetas
        h = _gabor_plane(data, sp, dim_a, dim_b, sigma, lambda, gamma, th, response, padding)
        if isnothing(out)
            out = h
        elseif pooling == "average"
            out .+= h
        elseif pooling == "max"
            out .= max.(out, h)
        else
            out .= min.(out, h)
        end
    end
    pooling == "average" && (out ./= length(thetas))
    return out
end

# Single-orientation Gabor response map in the plane (dim_a, dim_b).
function _gabor_plane(data::Array{Float64}, sp::Vector{Float64}, dim_a::Int, dim_b::Int,
    sigma::Float64, lambda::Float64, gamma::Float64, theta::Float64,
    response::String, padding::String)::Array{Float64}

    Na, Nb = size(data, dim_a), size(data, dim_b)
    Ma = iseven(Na) ? Na + 1 : Na
    Mb = iseven(Nb) ? Nb + 1 : Nb
    kernel = _gabor_kernel(Ma, Mb, sp[dim_a], sp[dim_b], sigma, lambda, gamma, theta)

    result = similar(data)
    # Bring the filtering plane to dims (1, 2); all other dims are looped over.
    perm = [dim_a, dim_b, [d for d in 1:ndims(data) if d != dim_a && d != dim_b]...]
    pdata = ndims(data) == 2 ? data : permutedims(data, perm)
    presult = ndims(data) == 2 ? result : similar(pdata)
    nslices = size(pdata, 3)  # 1 for 2D (trailing singleton)
    pdata3 = reshape(pdata, Na, Nb, :)
    presult3 = reshape(presult, Na, Nb, :)

    Threads.@threads for s in 1:size(pdata3, 3)
        slice = @view pdata3[:, :, s]
        padded = _pad_plane(slice, Ma ÷ 2, Mb ÷ 2, padding)
        conv = _conv2_valid(padded, kernel)
        @views presult3[:, :, s] .= _response(conv, response)
    end

    return ndims(data) == 2 ? presult : permutedims(presult, invperm(perm))
end

# Complex Gabor kernel on an Ma × Mb grid centred on the middle element. Offsets are physical.
function _gabor_kernel(Ma::Int, Mb::Int, sa::Float64, sb::Float64, sigma::Float64,
    lambda::Float64, gamma::Float64, theta::Float64)::Matrix{ComplexF64}
    c, s = cos(theta), sin(theta)
    ca, cb = (Ma + 1) ÷ 2, (Mb + 1) ÷ 2
    g = Matrix{ComplexF64}(undef, Ma, Mb)
    for j in 1:Mb, i in 1:Ma
        k1 = (i - ca) * sa
        k2 = (j - cb) * sb
        t1 = c * k1 + s * k2
        t2 = s * k1 - c * k2
        g[i, j] = exp(-(t1^2 + gamma^2 * t2^2) / (2 * sigma^2) + im * 2π * t1 / lambda)
    end
    return g
end

function _response(conv::Matrix{ComplexF64}, response::String)::Matrix{Float64}
    response == "modulus" && return abs.(conv)
    response == "real" && return real.(conv)
    response == "imaginary" && return imag.(conv)
    return angle.(conv)
end

# Pad a plane by `pa` / `pb` samples on each side of dims 1 / 2.
function _pad_plane(slice::AbstractMatrix{Float64}, pa::Int, pb::Int, padding::String)::Matrix{Float64}
    Na, Nb = size(slice)
    out = zeros(Float64, Na + 2pa, Nb + 2pb)
    for j in 1:(Nb+2pb), i in 1:(Na+2pa)
        ii = _pad_index(i - pa, Na, padding)
        jj = _pad_index(j - pb, Nb, padding)
        (ii == 0 || jj == 0) && continue
        out[i, j] = slice[ii, jj]
    end
    return out
end

# Map an out-of-range 1-based index into 1:N according to the padding mode (0 = zero fill).
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

# True 2D convolution; `padded` is the image padded by half the kernel size, output is image-sized.
function _conv2_valid(padded::Matrix{Float64}, kernel::Matrix{ComplexF64})::Matrix{ComplexF64}
    Ma, Mb = size(kernel)
    Na, Nb = size(padded, 1) - Ma + 1, size(padded, 2) - Mb + 1
    out = zeros(ComplexF64, Na, Nb)
    @inbounds for j in 1:Nb, i in 1:Na
        acc = zero(ComplexF64)
        for q in 1:Mb, p in 1:Ma
            # convolution: kernel flipped relative to correlation
            acc += padded[i+p-1, j+q-1] * kernel[Ma-p+1, Mb-q+1]
        end
        out[i, j] = acc
    end
    return out
end