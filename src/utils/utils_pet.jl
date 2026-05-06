using DICOM

"""
    sanitize(v)
    
    Sanitizes a value by removing any whitespace or empty arrays.
    
    # Arguments:
    - `v`: Any – the value to sanitize
    
    # Returns:
    - `String` – the sanitized value
"""
function sanitize(v)
    v === nothing && return ""
    s = (v isa AbstractArray || v isa Tuple) ? join(v) : string(v)
    return strip(s)
end

"""
    get_tag(dcm, tag)
    
    Gets the tag from the DICOM data.
    
    # Arguments:
    - `dcm`: DICOMData – the DICOM data
    - `tag`: Tuple{Int64, Int64} – the tag
    
    # Returns:
    - `Any` – the tag value
"""
function get_tag(dcm, tag)
    haskey(dcm, tag) || return nothing
    v = dcm[tag]
    (v === nothing || v == "" || (v isa AbstractArray && isempty(v))) && return nothing
    return v
end

"""
    parse_time(t_raw)
    
    Parses the time from the raw data.
    
    # Arguments:
    - `t_raw`: Any – the raw time data
    
    # Returns:
    - `Float64` – the time in seconds
"""
function parse_time(t_raw)
    t = sanitize(t_raw)
    isempty(t) && return nothing
    try
        h = parse(Float64, t[1:2])
        m = parse(Float64, t[3:4])
        s = parse(Float64, t[5:end])
        return h * 3600.0 + m * 60.0 + s
    catch
        return nothing
    end
end

"""
    parse_datetime_to_sec(dt_raw)
    
    Parses the datetime from the raw data.
    
    # Arguments:
    - `dt_raw`: Any – the raw datetime data
    
    # Returns:
    - `Float64` – the datetime in seconds
"""
function parse_datetime_to_sec(dt_raw)
    dt = sanitize(dt_raw)
    length(dt) < 14 && return nothing
    return parse_time(dt[9:end])
end
 
"""
    manuf_is(m, v)
    
    Checks if the manufacturer is equal to the given value.
    
    # Arguments:
    - `m`: String – the manufacturer
    - `v`: String – the value to compare
    
    # Returns:
    - `Bool` – true if the manufacturer is equal to the given value
"""
function manuf_is(m, v)
    occursin(uppercase(v), uppercase(sanitize(m)))
end

"""
    scalar_tag(d, tag)
    
    Gets the scalar tag from the DICOM data.
    
    # Arguments:
    - `d`: DICOMData – the DICOM data
    - `tag`: Tuple{Int64, Int64} – the tag
    
    # Returns:
    - `Float64` – the scalar tag value

"""
function scalar_tag(d, tag)
    v = get_tag(d, tag)
    v === nothing && return nothing
    if v isa AbstractArray{UInt8}
        s = strip(String(copy(v)))
        return tryparse(Float64, s)
    end
    return Float64(v isa AbstractArray ? first(v) : v)
end

"""
    get_dose(item)
    
    Gets the dose from the DICOM data.
    
    # Arguments:
    - `item`: DICOMData – the DICOM data
    
    # Returns:
    - `Float64` – the dose in MBq
"""
function get_dose(item)
    d = get_tag(item, (0x0018, 0x1074))
    d === nothing && return nothing
    D = Float64(d isa AbstractArray ? first(d) : d)
    return D < 1e4 ? D * 1e6 : D
end
"""
    get_tadm(item, t_acq_sec)
    
    Gets the time from the acquisition time.
    
    # Arguments:
    - `item`: DICOMData – the DICOM data
    - `t_acq_sec`: Float64 – the acquisition time in seconds
    
    # Returns:
    - `Float64` – the time in seconds
"""
function get_tadm(item, t_acq_sec)
    dt = get_tag(item, (0x0018, 0x1078))
    if dt !== nothing
        v = parse_datetime_to_sec(dt)
        if v !== nothing
            if t_acq_sec !== nothing && (v - t_acq_sec) > 3600.0
                v -= 86400.0
            end
            return v
        end
    end
    tm = get_tag(item, (0x0018, 0x1072))
    if tm !== nothing
        v = parse_time(tm)
        if v !== nothing
            if t_acq_sec !== nothing && (v - t_acq_sec) > 3600.0
                v -= 86400.0
            end
            return v
        end
    end
    return nothing
end
 
"""
    lbm_james128(W_kg, H_cm, sex)
    
    Calculates the lean body mass using the James formula.
    
    # Arguments:
    - `W_kg`: Float64 – the weight in kilograms
    - `H_cm`: Float64 – the height in centimeters
    - `sex`: String – the sex
    
    # Returns:
    - `Float64` – the lean body mass in kilograms
"""
function lbm_james128(W_kg, H_cm, sex)
    M = 1.10 * W_kg - 128.0 * (W_kg / H_cm)^2
    F = 1.07 * W_kg - 148.0 * (W_kg / H_cm)^2
    return (sex == "M" || sex == "1") ? M : (sex == "F" || sex == "2") ? F : (M + F) / 2.0
end
 
"""
    lbm_janma(W_kg, H_cm, sex)
    
    Calculates the lean body mass using the Janma formula.
    
    # Arguments:
    - `W_kg`: Float64 – the weight in kilograms
    - `H_cm`: Float64 – the height in centimeters
    - `sex`: String – the sex
    
    # Returns:
    - `Float64` – the lean body mass in kilograms
"""
function lbm_janma(W_kg, H_cm, sex)
    BMI = W_kg / (H_cm / 100.0)^2
    M = (9270.0 * W_kg) / (6680.0 + 216.0 * BMI)
    F = (9270.0 * W_kg) / (8780.0 + 244.0 * BMI)
    return (sex == "M" || sex == "1") ? M : (sex == "F" || sex == "2") ? F : (M + F) / 2.0
end

"""
    ibw(H_cm, sex)
    
    Calculates the ideal body weight using the Hume formula.
    
    # Arguments:
    - `H_cm`: Float64 – the height in centimeters
    - `sex`: String – the sex
    
    # Returns:
    - `Float64` – the ideal body weight in kilograms
"""
function ibw(H_cm, sex)
    M = 48.0 + 1.06 * (H_cm - 152.0)
    F = 45.5 + 0.91 * (H_cm - 152.0)
    return (sex == "M" || sex == "1") ? M : (sex == "F" || sex == "2") ? F : (M + F) / 2.0
end

"""
    bsa_dubois(W_kg, H_cm)
    
    Calculates the body surface area using the Dubois formula.
    
    # Arguments:
    - `W_kg`: Float64 – the weight in kilograms
    - `H_cm`: Float64 – the height in centimeters
    
    # Returns:
    - `Float64` – the body surface area
"""
function bsa_dubois(W_kg, H_cm)
    0.007184 * H_cm^0.725 * W_kg^0.425
end

"""
    t_ave(λ, T_s)
    
    Calculates the average time from the half-life and the scan duration.
    
    # Arguments:
    - `λ`: Float64 – the half-life
    - `T_s`: Float64 – the scan duration
    
    # Returns:
    - `Float64` – the average time
"""
function t_ave(λ, T_s)
    (1.0 / λ) * log(λ * T_s / (1.0 - exp(-λ * T_s)))
end

"""
    get_tref(d, λ, manufacturer)
    
    Gets the reference time from the DICOM data.
    
    # Arguments:
    - `d`: DICOMData – the DICOM data
    - `λ`: Float64 – the half-life
    - `manufacturer`: String – the manufacturer
    
    # Returns:
    - `Symbol` – the reference time mode (:admin, :start, or :none)
    - `Float64` – the reference time in seconds
"""
function get_tref(d, λ, manufacturer)
    dc        = sanitize(get_tag(d, (0x0054, 0x1102)))
    t_s_raw   = get_tag(d, (0x0008, 0x0031))
    t_acq_raw = get_tag(d, (0x0008, 0x0032))
    Δt_ms_raw = get_tag(d, (0x0054, 0x1300))
    T_ms_raw  = get_tag(d, (0x0018, 0x1242))
 
    t_s   = parse_time(t_s_raw)
    t_acq = parse_time(t_acq_raw)
    Δt    = Δt_ms_raw === nothing ? nothing : Float64(Δt_ms_raw isa AbstractArray ? first(Δt_ms_raw) : Δt_ms_raw) / 1000.0
    T_s   = T_ms_raw  === nothing ? nothing : Float64(T_ms_raw  isa AbstractArray ? first(T_ms_raw)  : T_ms_raw)  / 1000.0
 
    if dc == "ADMIN"
        return :admin, nothing
 
    elseif dc == "NONE"
        (t_acq === nothing || T_s === nothing || T_s <= 0) && return :error, nothing
        return :none, t_acq + t_ave(λ, T_s)
 
    else  # START (default)
        # 1) Siemens private tag (0071,1022)
        if manuf_is(manufacturer, "SIEMENS")
            sie = get_tag(d, (0x0071, 0x1022))
            if sie !== nothing
                raw = sie isa AbstractArray{UInt8} ? String(sie) : sanitize(sie)
                v = parse_datetime_to_sec(raw)
                v !== nothing && return :start, v
            end
        end

        # 2) GE private tag (0009,100D)
        if manuf_is(manufacturer, "GE")
            ge = get_tag(d, (0x0009, 0x100D))
            if ge !== nothing
                raw = ge isa AbstractArray{UInt8} ? String(ge) : sanitize(ge)
                v = parse_datetime_to_sec(raw)
                v !== nothing && return :start, v
            end
        end

        # 3) Tacq == Ts → usa direttamente t_acq (vale per SIEMENS, GE, PHILIPS)
        if manuf_is(manufacturer, "SIEMENS") || manuf_is(manufacturer, "GE") || manuf_is(manufacturer, "PHILIPS")
            if t_acq !== nothing && t_s !== nothing && abs(t_acq - t_s) < 1.0
                return :start, t_acq
            end
        end

        # 4) Fallback multi-bed Siemens/Philips (Tacq != Ts)
        if manuf_is(manufacturer, "SIEMENS") || manuf_is(manufacturer, "PHILIPS")
            if t_acq !== nothing && Δt !== nothing && T_s !== nothing && T_s > 0
                return :start, t_acq + t_ave(λ, T_s) - Δt
            end
        end

        # 5) Fallback multi-bed GE (Tacq != Ts)
        if manuf_is(manufacturer, "GE") && t_acq !== nothing && Δt !== nothing
            return :start, t_acq - Δt
        end

        # 6) Fallback: AcquisitionTime
        t_acq !== nothing && return :start, t_acq

        # 7) Fallback: SeriesTime
        t_s !== nothing && return :start, t_s

        return :error, nothing
    end
    
end
"""
    compute_slice_suv(d, units, suv_type, sex, W_kg, H_m,
                            D_adm, T_half, t_adm_sec, manufacturer)
    
    Computes the SUVbw value for a single PET slice.

    # Arguments:
    - `d`: DICOMData – the slice DICOM data
    - `units`: String – the SUV unit
    - `suv_type`: String – the SUV type
    - `sex`: String – the patient sex
    - `W_kg`: Float64 – the patient weight in kg
    - `H_m`: Float64 – the patient height in meters
    - `D_adm`: Float64 – the administered dose
    - `T_half`: Float64 – the half-life
    - `t_adm_sec`: Float64 – the acquisition time in seconds
    - `manufacturer`: String – the manufacturer

    # Returns:
    - `Float32` – the SUVbw value
    - `String` – the SUV type
"""
function compute_slice_suv(d, units, suv_type, sex, W_kg, H_m,
                            D_adm, T_half, t_adm_sec, manufacturer)

    m_slope = scalar_tag(d, (0x0028, 0x1053))
    b_inter = scalar_tag(d, (0x0028, 0x1052))
    (m_slope === nothing || b_inter === nothing) && return nothing, "error-rescale"
    abs(b_inter) > 1e-6 && return nothing, "error-nonzero-intercept"
    m_slope <= 0.0 && return nothing, "error-nonpositive-slope"

    P = Float64.(d[(0x7fe0, 0x0010)])
    U = m_slope .* P .+ b_inter

    H_cm = H_m * 100.0
    W_g  = W_kg >= 1000.0 ? W_kg : W_kg * 1000.0
    λ    = (T_half !== nothing && T_half > 0) ? log(2) / T_half : nothing

    if units == "GML"
        if suv_type ∈ ("", "BW")
            return Float32.(U), "SUVbw"
        elseif suv_type == "LBMJAMES128"
            f = lbm_james128(W_kg, H_cm, sex)
            return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_LBMjames"
        elseif suv_type == "LBMJANMA"
            f = lbm_janma(W_kg, H_cm, sex)
            return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_LBMjanma"
        elseif suv_type == "IBW"
            f = ibw(H_cm, sex)
            return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_IBW"
        else
            return nothing, "error-unknown-suvtype-$suv_type"
        end

    elseif units == "CM2ML"
        BSA = bsa_dubois(W_kg, H_cm)
        return Float32.(U .* (W_g / (BSA * 1e4))), "SUVbw_BSA"

    elseif units == "BQML" || units == "CNTS" || units == "CPS"

        if units == "CNTS"
            acsf = scalar_tag(d, (0x7053, 0x1009))
            if acsf !== nothing && acsf > 0
                U = U .* acsf
            else
                suvsf = scalar_tag(d, (0x7053, 0x1000))
                if suvsf !== nothing && suvsf > 0
                    return Float32.(U .* suvsf), "SUVbw_Philips"
                end
                return nothing, "error-CNTS-no-factor"
            end

        elseif units == "CPS"
            ci = sanitize(get_tag(d, (0x0028, 0x0051)))
            !occursin("DCAL", ci) && return nothing, "error-CPS-not-DCAL"

            px = get_tag(d, (0x0028, 0x0030))
            z  = scalar_tag(d, (0x0018, 0x0050))
            (px === nothing || z === nothing) && return nothing, "error-CPS-no-voxelsize"

            x = Float64(px isa AbstractArray ? first(px) : px)
            V = x^2 * z / 1000.0
            V <= 0 && return nothing, "error-CPS-zero-volume"

            U = U ./ V  # CPS → Bq/ml
        end

        # da qui BQML, CNTS (via ACSF) e CPS condividono la stessa logica
        λ === nothing && return nothing, "error-no-T½"
        dc_mode, t_ref = get_tref(d, λ, manufacturer)

        if dc_mode == :admin
            D = D_adm
        elseif dc_mode ∈ (:start, :none)
            t_ref    === nothing && return nothing, "error-no-tref"
            t_adm_sec === nothing && return nothing, "error-no-tadm"
            D_adm    === nothing && return nothing, "error-no-dose"
            Δt = t_ref - t_adm_sec
            D  = D_adm * exp(-λ * Δt)
        else
            return nothing, "error-tref"
        end

        return Float32.(U .* (W_g / D)), "SUVbw"
    end

    return nothing, "error-unit-$units"
end
