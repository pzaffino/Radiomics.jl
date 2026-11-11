# Shared helpers for image filters (wavelets and convolutional filters).

# Map an out-of-range 1-based index into 1:N according to the padding mode (0 = zero fill).
# Padding modes (IBSI 2 reference manual, sec. 2.2): "zero", "nearest", "periodic", "mirror".
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

# Filter `data` along `axis` with an odd-length kernel, handling the boundary via `padding`.
# Computes out[i] = Σ_t kernel[t] * data[i + t - r - 1] (correlation, kernel centred at r + 1).
# For a symmetric kernel this equals convolution; for a non-symmetric kernel (e.g. wavelets)
# pass `reverse(kernel)` to obtain a true convolution.
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