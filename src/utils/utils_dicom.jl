
"""
    slice_pos(d)

    Position of a slice along the normal of the image plane, used to sort a DICOM series.
    The normal is the cross product of the row and column direction cosines
    (`ImageOrientationPatient`); the position is the projection of
    `ImagePositionPatient` onto it. Falls back to `InstanceNumber` if the tags are missing.
"""
function slice_pos(d)
    if haskey(d, (0x0020, 0x0032)) && haskey(d, (0x0020, 0x0037))
        ipp = Float64.(collect(d[(0x0020, 0x0032)]))   # ImagePositionPatient
        iop = Float64.(collect(d[(0x0020, 0x0037)]))   # ImageOrientationPatient
        if length(ipp) == 3 && length(iop) == 6
            # normal = row direction × column direction
            nx = iop[2] * iop[6] - iop[3] * iop[5]
            ny = iop[3] * iop[4] - iop[1] * iop[6]
            nz = iop[1] * iop[5] - iop[2] * iop[4]
            # project the slice origin onto the normal
            return nx * ipp[1] + ny * ipp[2] + nz * ipp[3]
        end
    end
    # Fallback: InstanceNumber
    if haskey(d, (0x0020, 0x0013))
        return Float64(d[(0x0020, 0x0013)])
    end
    # No usable tag: ordering will be arbitrary
    return 0.0
end

"""
    get_pixel_measures(d)

    Returns `(pixel_spacing, slice_thickness)` from a DICOM dataset.
    Looks first at the top-level tags (`PixelSpacing`, `SliceThickness`) and then,
    for enhanced/multiframe files, inside `PixelMeasuresSequence` (0028,9110),
    which is nested in `SharedFunctionalGroupsSequence` (5200,9229) or in the
    first item of `PerFrameFunctionalGroupsSequence` (5200,9230).
    Returns `nothing` for any value that cannot be found.
"""
function get_pixel_measures(d)
    # Safe tag lookup: returns nothing if the tag is missing
    getval(ds, tag) = haskey(ds, tag) ? ds[tag] : nothing
    # Sequences are vectors of datasets: return the first item, or nothing
    first_item(seq) = (seq isa AbstractVector && !isempty(seq)) ? seq[1] : nothing

    pixel_spacing = getval(d, (0x0028, 0x0030))   # PixelSpacing
    slice_thickness = getval(d, (0x0018, 0x0050))   # SliceThickness

    # Fallback: PixelMeasuresSequence in the functional groups
    if pixel_spacing === nothing || slice_thickness === nothing
        for group_tag in ((0x5200, 0x9229), (0x5200, 0x9230))
            group = first_item(getval(d, group_tag))
            group === nothing && continue
            pms = first_item(getval(group, (0x0028, 0x9110)))
            pms === nothing && continue
            if pixel_spacing === nothing
                pixel_spacing = getval(pms, (0x0028, 0x0030))
            end
            if slice_thickness === nothing
                slice_thickness = getval(pms, (0x0018, 0x0050))
            end
            (pixel_spacing !== nothing && slice_thickness !== nothing) && break
        end
    end

    # Normalize types: PixelSpacing as Vector{Float64}, SliceThickness as Float64
    if pixel_spacing !== nothing
        pixel_spacing = Float64.(collect(pixel_spacing))
    end
    if slice_thickness !== nothing
        slice_thickness = Float64(slice_thickness isa AbstractArray ? first(slice_thickness) : slice_thickness)
    end

    return pixel_spacing, slice_thickness
end

"""
    extract_multiframe_pixel_slices(d, rows, cols, n_frames)

Returns a `Vector{Matrix{Float32}}` with one `rows × cols` matrix per frame,
read from the PixelData (7FE0,0010) of a multiframe DICOM dataset.
Handles PixelData returned either as a 3D array or as a flat vector.
"""
function extract_multiframe_pixel_slices(d, rows::Int, cols::Int, n_frames::Int)
    haskey(d, (0x7fe0, 0x0010)) || error("PixelData not found")
    px = d[(0x7fe0, 0x0010)]

    if px isa AbstractArray && ndims(px) == 3
        size(px, 3) == n_frames ||
            error("PixelData has $(size(px, 3)) frames, expected $n_frames")
        if size(px, 1) == rows && size(px, 2) == cols
            arr = px
        elseif size(px, 1) == cols && size(px, 2) == rows
            arr = permutedims(px, (2, 1, 3))   # (cols, rows, n) -> (rows, cols, n)
        else
            error("PixelData size $(size(px)) does not match rows=$rows, cols=$cols")
        end
    else
        # Flat vector: DICOM stores pixels row-major (columns vary fastest)
        length(px) == rows * cols * n_frames ||
            error("PixelData length $(length(px)) != rows*cols*n_frames")
        arr = permutedims(reshape(vec(px), cols, rows, n_frames), (2, 1, 3))
    end

    return [Float32.(arr[:, :, i]) for i in 1:n_frames]
end

# Reads a scalar value (or the first element) from a tag, with a default value.
function _scalar(ds, tag, default)
    (ds isa AbstractDict || hasmethod(haskey, Tuple{typeof(ds),typeof(tag)})) || return default
    haskey(ds, tag) || return default
    v = ds[tag]
    v = v isa AbstractArray ? first(v) : v
    v isa AbstractString ? parse(Float64, v) : Float64(v)
end

"""
    get_rescale(d; frame=nothing)

    Returns `(slope, intercept)`. Looks at top-level tags first; for multiframe files
    falls back on PixelValueTransformationSequence (0028,9145) in the per-frame
    or shared functional groups.
"""
function get_rescale(d; frame=nothing)
    if haskey(d, (0x0028, 0x1053)) || haskey(d, (0x0028, 0x1052))
        return _scalar(d, (0x0028, 0x1053), 1.0), _scalar(d, (0x0028, 0x1052), 0.0)
    end
    first_item(seq) = (seq isa AbstractVector && !isempty(seq)) ? seq[1] : nothing
    groups = Any[]
    if frame !== nothing && haskey(d, (0x5200, 0x9230))
        pf = d[(0x5200, 0x9230)]
        frame <= length(pf) && push!(groups, pf[frame])
    end
    haskey(d, (0x5200, 0x9229)) && push!(groups, first_item(d[(0x5200, 0x9229)]))
    for g in groups
        g === nothing && continue
        haskey(g, (0x0028, 0x9145)) || continue
        pvt = first_item(g[(0x0028, 0x9145)])
        pvt === nothing && continue
        return _scalar(pvt, (0x0028, 0x9153), 1.0), _scalar(pvt, (0x0028, 0x9152), 0.0)
    end
    return 1.0, 0.0
end