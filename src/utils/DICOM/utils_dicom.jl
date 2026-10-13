
using Dates

include("normalize_IBSI_SUV.jl")

"""
    _read_dicom_suv(path, roi_slices)

    Reads a PET DICOM series and returns it normalized to SUVbw (IBSI).
    The SUV is computed only on the slices in `roi_slices` (all frames for multiframe).

    # Arguments:
    - `path`: The path to the image.
    - `roi_slices`: The slices to normalize.

    # Returns:
    - `MedImage`: The medical image.
"""
function _read_dicom_suv(path, roi_slices)
    (isdir(path) || endswith(lowercase(path), ".dcm")) ||
        error("suv_ibsi_normalization=true requires a DICOM PET series: $path")
    isempty(roi_slices) && error("Mask is empty")

    base = _read_dicom(path)                       # geometry and spacing
    dcms, nf = _load_dicom_series(path)
    d0 = dcms[1]

    sanitize(get_tag(d0, (0x0008, 0x0060))) == "PT" ||
        error("suv_ibsi_normalization=true requires a PET series (Modality PT): $path")

    units = sanitize(get_tag(d0, (0x0054, 0x1001)))
    isempty(units) && (units = get_units_from_rwv(d0))
    suv_type = sanitize(get_tag(d0, (0x0054, 0x1006)))
    sex = sanitize(get_tag(d0, (0x0010, 0x0040)))
    manuf = sanitize(get_tag(d0, (0x0008, 0x0070)))

    W_raw = scalar_tag(d0, (0x0010, 0x1030))
    (W_raw === nothing || W_raw <= 0) && error("Patient's Weight assente o non positivo")
    W_kg = W_raw >= 1000.0 ? W_raw / 1000.0 : W_raw
    H_m = something(scalar_tag(d0, (0x0010, 0x1020)), 0.0)

    r_seq_0 = get_tag(d0, (0x0054, 0x0016))
    rp_item_0 = r_seq_0 !== nothing ? r_seq_0[1] : d0
    D_adm = get_dose(rp_item_0)
    T_half = something(scalar_tag(rp_item_0, (0x0018, 0x1075)), 0.0)

    rows = Int(d0[(0x0028, 0x0010)])
    cols = Int(d0[(0x0028, 0x0011)])
    is_multiframe = length(dcms) == 1 && nf > 1
    nz = is_multiframe ? nf : length(dcms)
    maximum(roi_slices) <= nz || error("Mask and PET have different number of slices")

    function suv_of(d)
        r_seq = get_tag(d, (0x0054, 0x0016))
        rp_item = r_seq !== nothing ? r_seq[1] : d
        acq_date = parse_date(get_tag(d, (0x0008, 0x0022)))
        t_acq_tod = parse_time(get_tag(d, (0x0008, 0x0032)))
        t_acq_dt = combine_date_time(acq_date !== nothing ? acq_date : Date(2000, 1, 1), t_acq_tod)
        t_adm, t_adm_q = get_tadm(rp_item, t_acq_dt)
        return compute_slice_suv(d, units, suv_type, sex, W_kg, H_m, D_adm, T_half,
            t_adm, t_adm_q, manuf)
    end

    suv_vol = zeros(Float32, rows, cols, nz)
    if is_multiframe
        pix = extract_multiframe_pixel_slices(d0, rows, cols, nf)
        for i in 1:nf
            res, reason = suv_of(FrameView(d0, i, pix[i]))
            res === nothing && error("SUV normalization failed on frame $i: $reason (units=$units, suv_type=$suv_type)")
            suv_vol[:, :, i] = res
        end
    else
        for i in roi_slices
            res, reason = suv_of(dcms[i])
            res === nothing && error("SUV normalization failed on slice $i: $reason (units=$units, suv_type=$suv_type)")
            suv_vol[:, :, i] = res
        end
    end

    data = permutedims(suv_vol, (2, 1, 3))         # (cols, rows, slices), come _read_dicom
    return MedImage(data, base.spacing, base.origin, base.direction)
end

"""
    _load_dicom_series(path)

    Loads a DICOM series (folder or single file), checks it is a single series
    and sorts the slices by position. Returns `(dcms, n_frames)`.

    # Arguments:
    - `path`: The path to the image.

    # Returns:
    - `dcms`: The list of DICOM images.
    - `n_frames`: The number of frames.
"""
function _load_dicom_series(path)
    dcms = isdir(path) ? DICOM.dcmdir_parse(path) : [DICOM.dcm_parse(path)]
    filter!(d -> haskey(d, (0x7fe0, 0x0010)), dcms)
    isempty(dcms) && error("No DICOM images found in $path")
    length(unique(d[(0x0020, 0x000e)] for d in dcms)) == 1 ||
        error("Folder contains more than one DICOM series: $path")
    nf = Int(_scalar(dcms[1], (0x0028, 0x0008), 1))
    length(dcms) > 1 && nf > 1 &&
        error("Multiple multiframe DICOM files not supported: $path")
    sort!(dcms; by=slice_pos)
    return dcms, nf
end

"""
    _scalar(ds, tag, default)
    
    Reads a scalar value (or the first element) from a tag, with a default value.
    
    # Arguments:
    - `ds`: The DICOM dataset.
    - `tag`: The tag to read.
    - `default`: The default value.
    
    # Returns:
    - `_scalar`: The scalar value.
"""
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
        return _scalar(pvt, (0x0028, 0x1053), 1.0), _scalar(pvt, (0x0028, 0x1052), 0.0)
    end

    @warn "No slope/intercept find, use (1.0, 0.0)" frame
    return 1.0, 0.0
end