# Julia side of the Python bridge: converts Python inputs to the types
# Radiomics.jl expects and converts the result back to native Python objects.
module RadiomicsJLBridge

using PythonCall
import Radiomics

# ---------- Python -> Julia ----------
_arr(x::AbstractString) = String(x)
_arr(x::Py) = Array(PyArray(x))
_arr(x::AbstractArray) = Array(x)

_spacing(::Nothing) = nothing
_spacing(x) = Float64[Float64(v) for v in _arr(x)]

_labels(::Nothing) = nothing
_labels(x::Integer) = Int(x)
_labels(x) = Int[Int(v) for v in _arr(x)]

_slices(::Nothing) = nothing
function _slices(x)
    m = _arr(x)                      # n×2 matrix (plane, slice)
    return Tuple{Int,Int}[(Int(m[i, 1]), Int(m[i, 2])) for i in axes(m, 1)]
end

_features(s::AbstractString) = isempty(s) ? Symbol[] : Symbol[Symbol(f) for f in split(s, ",")]

# ---------- Julia -> Python ----------
const _np = Ref{Py}()
_numpy() = isassigned(_np) ? _np[] : (_np[] = pyimport("numpy"))

topy(::Nothing) = pybuiltins.None
topy(x::Bool) = Py(x)
topy(x::Integer) = Py(Int(x))
topy(x::AbstractFloat) = Py(Float64(x))
topy(x::Union{AbstractString,Symbol}) = Py(String(x))
topy(x::Tuple) = pytuple([topy(v) for v in x])
function topy(x::AbstractDict)
    d = pydict()
    for (k, v) in x
        d[topy(k)] = topy(v)
    end
    return d
end
function topy(x::AbstractArray)
    if ndims(x) == 1 || !(eltype(x) <: Number)
        return pylist([topy(v) for v in x])          # small vectors (sizes, spacing) -> list
    end
    return _numpy().array(Py(Array(x)); copy=true)   # matrices -> numpy (copy, owns its memory)
end
topy(x) = Py(x)

function extract(img, mask, spacing, features, labels, slices_2d, n_bins, bin_width,
                 weighting_norm, keep_largest_only, get_raw_matrices, features_std,
                 use_gpu, suv_ibsi_normalization, verbose)
    res = Radiomics.extract_radiomic_features(
        _arr(img), _arr(mask), _spacing(spacing);
        features=_features(features),
        labels=_labels(labels),
        n_bins=n_bins,
        bin_width=bin_width,
        weighting_norm=weighting_norm,
        keep_largest_only=keep_largest_only,
        get_raw_matrices=get_raw_matrices,
        features_std=features_std,
        slices_2d=_slices(slices_2d),
        use_gpu=use_gpu,
        suv_ibsi_normalization=suv_ibsi_normalization,
        verbose=verbose)
    return topy(res)
end

version() = string(pkgversion(Radiomics))

end # module
