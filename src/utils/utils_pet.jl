using DICOM
using Dates

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

function get_pixel_measures(d)
    ps_top = get_tag(d, (0x0028, 0x0030))
    if ps_top !== nothing
        ps = Float64.(collect(ps_top))
        st = scalar_tag(d, (0x0018, 0x0050))
        return ps, st
    end

    shared = get_tag(d, (0x5200, 0x9229))
    if shared !== nothing && !isempty(shared)
        pm = _pixel_measures_from_group(shared[1])
        pm !== nothing && return pm
    end

    per_frame = get_tag(d, (0x5200, 0x9230))
    if per_frame !== nothing && !isempty(per_frame)
        pm = _pixel_measures_from_group(per_frame[1])
        pm !== nothing && return pm
    end

    return nothing, nothing
end

function _pixel_measures_from_group(item)
    pms = get_tag(item, (0x0028, 0x9110))
    (pms === nothing || isempty(pms)) && return nothing

    entry = pms[1]
    ps_raw = get_tag(entry, (0x0028, 0x0030))
    ps_raw === nothing && return nothing

    ps = Float64.(collect(ps_raw))
    st = scalar_tag(entry, (0x0018, 0x0050))
    return ps, st
end

"""
    parse_time(t_raw)
    
    Parses the time-of-day (no date) from the raw data.
    
    # Arguments:
    - `t_raw`: Any – the raw time data
    
    # Returns:
    - `Float64` – the time in seconds since midnight
"""
function parse_time(t_raw)
    t = sanitize(t_raw)
    isempty(t) && return nothing
    try
        h = parse(Float64, t[1:2])
        m = parse(Float64, t[3:4])
        sec_str = t[5:end]
        tz_idx = findfirst(c -> c == '+' || c == '-', sec_str)
        if tz_idx !== nothing
            sec_str = sec_str[1:tz_idx-1]
        end
        s = isempty(sec_str) ? 0.0 : parse(Float64, sec_str)
        return h * 3600.0 + m * 60.0 + s
    catch
        return nothing
    end
end

"""
    _PLACEHOLDER_DATES

    Date DICOM (formato YYYYMMDD) comunemente usate dai tool di
    anonimizzazione per sostituire una data reale. Se una data DA/DT
    coincide con uno di questi valori, viene trattata come ASSENTE
    (non affidabile), non come una data valida.

    ATTENZIONE: euristica non verificata contro il tool di
    anonimizzazione realmente usato per generare i DRO — verificare e
    aggiornare questo set se il comportamento osservato differisce.
"""
const _PLACEHOLDER_DATES = Set(["00010101", "18000101", "19000101"])

"""
    parse_date(d_raw)

    Parses a DICOM DA value (YYYYMMDD) into a `Date`. Returns `nothing`
    if missing, malformed, or matching a known anonymization placeholder
    (see `_PLACEHOLDER_DATES`).
"""
function parse_date(d_raw)
    d = sanitize(d_raw)
    length(d) < 8 && return nothing
    d8 = d[1:8]
    d8 ∈ _PLACEHOLDER_DATES && return nothing
    try
        y = parse(Int, d8[1:4])
        mo = parse(Int, d8[5:6])
        day = parse(Int, d8[7:8])
        return Date(y, mo, day)
    catch
        return nothing
    end
end

"""
    parse_datetime(dt_raw)

    Parses a DICOM DT value (YYYYMMDDHHMMSS[.ffffff][&ZZXX]) into a
    full `DateTime` (data E ora) — a differenza della vecchia
    `parse_datetime_to_sec`, che scartava la data. Ritorna `nothing`
    se assente, malformato, o se la data coincide con un placeholder
    di anonimizzazione (vedi `_PLACEHOLDER_DATES`): in tal caso il
    chiamante deve considerare il valore come se fosse disponibile
    solo l'ora, non un riferimento datato affidabile.
"""
function parse_datetime(dt_raw)
    dt = sanitize(dt_raw)
    length(dt) < 14 && return nothing
    dt[1:8] ∈ _PLACEHOLDER_DATES && return nothing
    try
        y = parse(Int, dt[1:4])
        mo = parse(Int, dt[5:6])
        day = parse(Int, dt[7:8])
        h = parse(Int, dt[9:10])
        mi = parse(Int, dt[11:12])
        sec_str = dt[13:end]
        tz_idx = findfirst(c -> c == '+' || c == '-', sec_str)
        tz_idx !== nothing && (sec_str = sec_str[1:tz_idx-1])
        s = isempty(sec_str) ? 0.0 : parse(Float64, sec_str)
        return DateTime(y, mo, day, h, mi, 0) + Millisecond(round(Int, s * 1000))
    catch
        return nothing
    end
end

"""
    parse_datetime_to_sec(dt_raw)

    [LEGACY] Ritorna solo la componente oraria (secondi nel giorno) di
    un valore DT, scartando la data. Non più usata internamente dal
    pacchetto (sostituita da `parse_datetime`); mantenuta solo per
    compatibilità con eventuale codice esterno che la richiami.
"""
function parse_datetime_to_sec(dt_raw)
    dt = sanitize(dt_raw)
    length(dt) < 14 && return nothing
    return parse_time(dt[9:end])
end

"""
    combine_date_time(date, time_sec)

    Combina una `Date` e un'ora-nel-giorno (secondi, come ritornato da
    `parse_time`) in un `DateTime` completo. Ritorna `nothing` se uno
    dei due argomenti manca.
"""
function combine_date_time(date, time_sec)
    (date === nothing || time_sec === nothing) && return nothing
    return DateTime(date) + Millisecond(round(Int, time_sec * 1000))
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
    D < 0.0 && return nothing  # dose negativa trattata come assente → "error-no-dose"
    return D < 1e4 ? D * 1e6 : D
end

"""
    get_tadm(item, t_acq_dt)

    Determina l'orario di somministrazione come `DateTime` pieno
    (data + ora), risolvendo l'ambiguità multi-giorno che affliggeva
    la vecchia implementazione basata solo su "secondi nel giorno".

    Ordine di risoluzione:
      1) Radiopharmaceutical Start DateTime (0018,1078) con data valida
         (non placeholder di anonimizzazione) → fonte più affidabile,
         usata direttamente. `qualified = true`.
      2) Se (1) è assente o con data non valida, si tenta comunque di
         estrarne la sola componente oraria e la si combina con la
         data di `t_acq_dt` (assunzione "stesso giorno
         dell'acquisizione", con correzione di un giorno se l'orario
         risultante precede l'acquisizione di più di un'ora).
         `qualified = false`.
      3) Fallback finale: Radiopharmaceutical Start Time (0018,1072),
         stessa logica del punto 2. `qualified = false`.

    # Arguments:
    - `item`: DICOMData – il dataset (o RadiopharmaceuticalInformationSequence item)
    - `t_acq_dt`: Union{DateTime,Nothing} – l'orario di acquisizione, usato
      come riferimento di data per i fallback non qualificati

    # Returns:
    - `Union{DateTime,Nothing}` – l'orario di somministrazione
    - `Bool` – `true` se la data è nota con certezza, `false` se assunta
"""
function get_tadm(item, t_acq_dt::Union{DateTime,Nothing})
    acq_date = t_acq_dt === nothing ? nothing : Date(t_acq_dt)

    function _same_day_fallback(time_of_day)
        (time_of_day === nothing || acq_date === nothing) && return nothing
        v = DateTime(acq_date) + Millisecond(round(Int, time_of_day * 1000))
        if t_acq_dt !== nothing && Dates.value(v - t_acq_dt) > 3_600_000  # > 1h dopo l'acquisizione
            v -= Day(1)
        end
        return v
    end

    dt_raw = get_tag(item, (0x0018, 0x1078))
    if dt_raw !== nothing
        v = parse_datetime(dt_raw)
        v !== nothing && return v, true

        # tag presente ma data assente/anonimizzata: proviamo comunque
        # a recuperarne la sola componente oraria
        raw = sanitize(dt_raw)
        if length(raw) >= 14
            tod = parse_time(raw[9:end])
            v2 = _same_day_fallback(tod)
            v2 !== nothing && return v2, false
        end
    end

    tm = get_tag(item, (0x0018, 0x1072))
    if tm !== nothing
        tod = parse_time(tm)
        v = _same_day_fallback(tod)
        v !== nothing && return v, false
    end

    return nothing, false
end

"""
    lbm_james128(W_kg, H_cm, sex)
    
    Calculates the lean body mass using the James formula.
"""
function lbm_james128(W_kg, H_cm, sex)
    M = 1.10 * W_kg - 128.0 * (W_kg / H_cm)^2
    F = 1.07 * W_kg - 148.0 * (W_kg / H_cm)^2
    return (sex == "M" || sex == "1") ? M : (sex == "F" || sex == "2") ? F : (M + F) / 2.0
end

"""
    lbm_janma(W_kg, H_cm, sex)
    
    Calculates the lean body mass using the Janma formula.
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
"""
function ibw(H_cm, sex)
    M = 48.0 + 1.06 * (H_cm - 152.0)
    F = 45.5 + 0.91 * (H_cm - 152.0)
    return (sex == "M" || sex == "1") ? M : (sex == "F" || sex == "2") ? F : (M + F) / 2.0
end

"""
    bsa_dubois(W_kg, H_cm)
    
    Calculates the body surface area using the Dubois formula.
"""
function bsa_dubois(W_kg, H_cm)
    0.007184 * H_cm^0.725 * W_kg^0.425
end

"""
    t_ave(λ, T_s)
    
    Calculates the average time from the half-life and the scan duration.
"""
function t_ave(λ, T_s)
    (1.0 / λ) * log(λ * T_s / (1.0 - exp(-λ * T_s)))
end

const _LONG_UPTAKE_HALFLIFE_SEC = 24 * 3600.0

"""
    _MAX_PLAUSIBLE_DECAY_SEC

    Soglia (in secondi) sul valore assoluto di `Δt = t_ref - t_adm` usato
    nel calcolo del decadimento. Oltre questa soglia, `exp(-λΔt)` va in
    overflow o underflow a seconda del segno, producendo silenziosamente
    SUV = 0 (tutta l'immagine) o SUV = Inf (che a sua volta manda in
    crash le fasi successive, es. il binning delle feature).

    30 giorni è generoso rispetto a qualunque radionuclide/protocollo
    clinico plausibile (anche imaging ritardato multi-giorno con
    radionuclidi a emivita lunga come Zr-89/I-124 resta ben sotto
    questa soglia), ma abbastanza stretto da intercettare un mismatch
    di epoca (es. una data anonimizzata non riconosciuta da
    `_PLACEHOLDER_DATES`, che può introdurre scarti di decenni tra
    `t_ref` e `t_adm`).

    Nota: questa guardia è una rete di sicurezza generica, non sostituisce
    la necessità di tenere `_PLACEHOLDER_DATES` aggiornato rispetto alle
    convenzioni di anonimizzazione realmente usate — trasforma però un
    bug di quel tipo da "risultato numerico silenziosamente sbagliato o
    crash a valle" a "errore esplicito e diagnosticabile".
"""
const _MAX_PLAUSIBLE_DECAY_SEC = 30 * 24 * 3600.0

"""
    get_tref(d, λ, manufacturer)
    
    Gets the reference time from the DICOM data, come `DateTime` pieno.
    
    # Arguments:
    - `d`: DICOMData – the DICOM data (o una `FrameView` per Enhanced/MultiFrame)
    - `λ`: Float64 – the decay constant
    - `manufacturer`: String – the manufacturer

    # Returns:
    - `Symbol` – the reference time mode (:admin, :start, :none, or :error)
    - `Union{DateTime,Nothing}` – the reference time

    # Note (fix Enhanced — Frame Reference DateTime):
    - Per i file Enhanced/MultiFrame, `(0018,9151) Frame Reference DateTime`
      (dentro `FrameContentSequence`, instradato per-frame da `FrameView`)
      è un tempo di riferimento ASSOLUTO fornito direttamente dallo
      standard DICOM per questo scopo — quando presente e valido, ha
      priorità su qualunque ricostruzione Δt/T_s/vendor-specifica pensata
      per il classico single-frame (che per l'Enhanced non si applica:
      i tag legacy `DecayCorrection`/`AcquisitionDate`/`ActualFrameDuration`
      non sono tipicamente popolati a livello a cui `get_tref` li cerca).

    # Note (fix vendor-gating):
    - La correzione Δt/T_s (Frame Reference Time / Actual Frame Duration)
      per il caso multi-bed (Tacq≠Ts) non è più gated sul manufacturer:
      si applica a QUALUNQUE vendor tranne GE (che ha una convenzione nota
      e diversa, senza il termine `t_ave`), quando Δt e T_s sono presenti
      e validi. Per SIEMENS/PHILIPS l'assenza di questi dati resta un
      errore esplicito (comportamento invariato); per vendor sconosciuti
      senza questi tag si degrada al fallback generico (SeriesTime) anziché
      fallire, dato che non c'è garanzia che li popolino.
"""
function get_tref(d, λ, manufacturer)
    dc = sanitize(get_tag(d, (0x0054, 0x1102)))

    s_date = parse_date(get_tag(d, (0x0008, 0x0021)))  # SeriesDate
    acq_date = parse_date(get_tag(d, (0x0008, 0x0022)))  # AcquisitionDate
    t_s_tod = parse_time(get_tag(d, (0x0008, 0x0031))) # SeriesTime
    t_acq_tod = parse_time(get_tag(d, (0x0008, 0x0032))) # AcquisitionTime

    ref_date = acq_date !== nothing ? acq_date : (s_date !== nothing ? s_date : Date(2000, 1, 1))
    t_s = combine_date_time(s_date !== nothing ? s_date : ref_date, t_s_tod)
    t_acq = combine_date_time(acq_date !== nothing ? acq_date : ref_date, t_acq_tod)

    Δt_ms_raw = get_tag(d, (0x0054, 0x1300))
    T_ms_raw = get_tag(d, (0x0018, 0x1242))
    Δt = Δt_ms_raw === nothing ? nothing : Float64(Δt_ms_raw isa AbstractArray ? first(Δt_ms_raw) : Δt_ms_raw) / 1000.0
    T_s = T_ms_raw === nothing ? nothing : Float64(T_ms_raw isa AbstractArray ? first(T_ms_raw) : T_ms_raw) / 1000.0

    if dc == "ADMIN"
        return :admin, nothing

    elseif dc == "NONE"
        (t_acq === nothing || T_s === nothing || T_s <= 0) && return :error, nothing
        return :none, t_acq + Millisecond(round(Int, t_ave(λ, T_s) * 1000))

    else  # START (default)

        # 0) Enhanced/MultiFrame: Frame Reference DateTime (0018,9151) è già
        #    un riferimento assoluto per-frame — priorità massima quando
        #    disponibile e valido (non placeholder di anonimizzazione).
        frdt_raw = get_tag(d, (0x0018, 0x9151))
        if frdt_raw !== nothing
            frdt = parse_datetime(frdt_raw)
            frdt !== nothing && return :start, frdt
        end

        # 1) Siemens private tag (0071,1022)
        if manuf_is(manufacturer, "SIEMENS")
            sie = get_tag(d, (0x0071, 0x1022))
            if sie !== nothing
                raw = sie isa AbstractArray{UInt8} ? String(sie) : sanitize(sie)
                v = parse_datetime(raw)
                v !== nothing && return :start, v
            end
        end

        # 2) GE private tag (0009,100D)
        if manuf_is(manufacturer, "GE")
            ge = get_tag(d, (0x0009, 0x100D))
            if ge !== nothing
                raw = ge isa AbstractArray{UInt8} ? String(ge) : sanitize(ge)
                v = parse_datetime(raw)
                v !== nothing && return :start, v
            end
        end

        known_vendor = manuf_is(manufacturer, "SIEMENS") || manuf_is(manufacturer, "GE") || manuf_is(manufacturer, "PHILIPS")
        same_time = t_acq !== nothing && t_s !== nothing && abs(Dates.value(t_acq - t_s)) < 1000.0

        # 3) Tacq == Ts → nessun offset multi-bed da correggere: usa
        #    direttamente t_acq. Generalizzato a qualunque vendor (non
        #    c'è nulla di vendor-specifico da correggere se non c'è
        #    offset).
        same_time && return :start, t_acq

        # 4) Tacq != Ts (multi-bed): formula generica (t_acq + t_ave(λ,T_s) - Δt),
        #    valida per qualunque vendor tranne GE (vedi punto 5). Prima era
        #    gated su SIEMENS/PHILIPS: ora si applica anche a vendor
        #    sconosciuti/conformi allo standard quando Δt e T_s sono
        #    presenti e validi. Per SIEMENS/PHILIPS l'assenza di questi dati
        #    resta un errore esplicito (comportamento invariato).
        if !manuf_is(manufacturer, "GE") && t_acq !== nothing && t_s !== nothing
            if Δt !== nothing && T_s !== nothing && T_s > 0
                return :start, t_acq + Millisecond(round(Int, (t_ave(λ, T_s) - Δt) * 1000))
            elseif known_vendor
                return :error, nothing
            end
            # vendor sconosciuto senza Δt/T_s → si scende al fallback generico (punto 6)
        end

        # 5) Fallback multi-bed GE (Tacq != Ts): convenzione nota e diversa,
        #    senza termine t_ave.
        if manuf_is(manufacturer, "GE") && t_acq !== nothing && t_s !== nothing
            Δt === nothing && return :error, nothing
            return :start, t_acq - Millisecond(round(Int, Δt * 1000))
        end

        # 6) Fallback generico: preferisci SeriesTime
        t_s !== nothing && return :start, t_s

        # 7) Ultimo fallback: AcquisitionTime
        t_acq !== nothing && return :start, t_acq

        return :error, nothing
    end

end

"""
    compute_slice_suv(d, units, suv_type, sex, W_kg, H_m,
                            D_adm, T_half, t_adm, t_adm_qualified, manufacturer)
    
    Computes the SUVbw value for a single PET slice.

    # Note (fix — guardia di plausibilità su Δt):
    - Prima di usare `Δt = t_ref - t_adm` nel calcolo del decadimento,
      si verifica `abs(Δt) <= _MAX_PLAUSIBLE_DECAY_SEC`. Senza questa
      guardia, un mismatch di epoca tra `t_ref` e `t_adm` (es. per una
      data di anonimizzazione non riconosciuta da `_PLACEHOLDER_DATES`)
      produce `exp(-λΔt)` in overflow o underflow, e quindi SUV
      silenziosamente pari a 0 (tutta l'immagine) o Inf (che manda in
      crash le fasi successive, es. il binning delle feature) —
      invece di un errore esplicito e diagnosticabile.
"""
function compute_slice_suv(d, units, suv_type, sex, W_kg, H_m,
    D_adm, T_half, t_adm, t_adm_qualified, manufacturer)

    m_slope = scalar_tag(d, (0x0028, 0x1053))
    b_inter = scalar_tag(d, (0x0028, 0x1052))
    (m_slope === nothing || b_inter === nothing) && return nothing, "error-rescale"
    abs(b_inter) > 1e-6 && return nothing, "error-nonzero-intercept"
    m_slope <= 0.0 && return nothing, "error-nonpositive-slope"

    P = Float64.(d[(0x7fe0, 0x0010)])
    U = m_slope .* P .+ b_inter

    H_cm = H_m * 100.0
    W_g = W_kg >= 1000.0 ? W_kg : W_kg * 1000.0
    λ = (T_half !== nothing && T_half > 0) ? log(2) / T_half : nothing

    if units == "GML"
        if suv_type ∈ ("", "BW")
            return Float32.(U), "SUVbw"
        elseif suv_type ∈ ("LBMJAMES128", "LBMJANMA", "IBW")
            H_m <= 0.0 && return nothing, "error-no-height"
            sex ∈ ("M", "F", "O", "1", "2") || return nothing, "error-invalid-sex"
            if suv_type == "LBMJAMES128"
                f = lbm_james128(W_kg, H_cm, sex)
                return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_LBMjames"
            elseif suv_type == "LBMJANMA"
                f = lbm_janma(W_kg, H_cm, sex)
                return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_LBMjanma"
            else # IBW
                f = ibw(H_cm, sex)
                return Float32.(U .* (W_g / (f * 1e3))), "SUVbw_IBW"
            end
        else
            return nothing, "error-unknown-suvtype-$suv_type"
        end

    elseif units == "CM2ML"
        H_m <= 0.0 && return nothing, "error-no-height"
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
            z = scalar_tag(d, (0x0018, 0x0050))
            (px === nothing || z === nothing) && return nothing, "error-CPS-no-voxelsize"

            x = Float64(px isa AbstractArray ? first(px) : px)
            V = x^2 * z / 1000.0
            V <= 0 && return nothing, "error-CPS-zero-volume"

            U = U ./ V  # CPS → Bq/ml
        end

        λ === nothing && return nothing, "error-no-T½"
        dc_mode, t_ref = get_tref(d, λ, manufacturer)

        if dc_mode == :admin
            D_adm === nothing && return nothing, "error-no-dose"
            D = D_adm
        elseif dc_mode ∈ (:start, :none)
            t_ref === nothing && return nothing, "error-no-tref"
            t_adm === nothing && return nothing, "error-no-tadm"
            D_adm === nothing && return nothing, "error-no-dose"
            if !t_adm_qualified && T_half !== nothing && T_half > _LONG_UPTAKE_HALFLIFE_SEC
                return nothing, "error-tadm-not-date-qualified"
            end
            Δt = Dates.value(t_ref - t_adm) / 1000.0
            abs(Δt) > _MAX_PLAUSIBLE_DECAY_SEC && return nothing, "error-decay-time-implausible"
            D = D_adm * exp(-λ * Δt)
        else
            return nothing, "error-tref"
        end

        return Float32.(U .* (W_g / D)), "SUVbw"
    end

    return nothing, "error-unit-$units"
end

"""
    FrameView

An adapter that exposes a single frame of an Enhanced PET DICOM file
(MultiFrame) using the same interface (`haskey`, `getindex`) as a
single-frame DICOMData object, allowing `get_tag`, `compute_slice_suv`,
and `get_tref` to be reused WITHOUT MODIFICATION for MultiFrame data.

For each requested tag, the search order is:
  1) PixelData (0x7fe0,0x0010) -> returns the pre-extracted 2D slice for
     this frame (`pixel_slice`), not the entire 3D block.
  2) Within PerFrameFunctionalGroupsSequence[frame_idx], inside the
     known sub-sequences (PixelValueTransformationSequence, FrameContentSequence).
  3) Within SharedFunctionalGroupsSequence[1], same sub-sequences.
  4) Fallback: the top-level tag on the original dataset `d0`.

# Note (fix — Frame Reference DateTime):
`(0x0018, 0x9151)` (Frame Reference DateTime) è stato aggiunto a
`_FRAME_SUBSEQUENCES`, instradato verso `FrameContentSequence`
(0x0020, 0x9111): prima non c'era alcun mapping per questo tag, quindi
`get_tag(fv, (0x0018,0x9151))` cadeva sempre sul fallback top-level
(`fv.d0[tag]`), dove per un Enhanced PET questo tag non esiste —
risultando sempre in `nothing` anche quando il valore per-frame era
presente e valido.
"""
struct FrameView
    d0::Any
    frame_idx::Int
    pixel_slice::Array{Float64,2}
end

const _FRAME_SUBSEQUENCES = Dict(
    (0x0028, 0x1053) => (0x0028, 0x9145),  # RescaleSlope <- PixelValueTransformationSequence
    (0x0028, 0x1052) => (0x0028, 0x9145),  # RescaleIntercept <- PixelValueTransformationSequence
    (0x0054, 0x1300) => (0x0020, 0x9111),  # FrameReferenceTime <- FrameContentSequence
    (0x0018, 0x9151) => (0x0020, 0x9111),  # FrameReferenceDateTime <- FrameContentSequence (fix)
)

function _search_functional_group(item, tag)
    subseq_tag = get(_FRAME_SUBSEQUENCES, tag, nothing)
    subseq_tag === nothing && return nothing
    haskey(item, subseq_tag) || return nothing
    subseq = item[subseq_tag]
    (subseq === nothing || isempty(subseq)) && return nothing
    entry = subseq[1]
    haskey(entry, tag) || return nothing
    return entry[tag]
end

function Base.haskey(fv::FrameView, tag::Tuple{UInt16,UInt16})
    tag == (0x7fe0, 0x0010) && return true

    pf = haskey(fv.d0, (0x5200, 0x9230)) ? fv.d0[(0x5200, 0x9230)] : nothing
    if pf !== nothing && fv.frame_idx <= length(pf)
        _search_functional_group(pf[fv.frame_idx], tag) !== nothing && return true
    end

    shared = haskey(fv.d0, (0x5200, 0x9229)) ? fv.d0[(0x5200, 0x9229)] : nothing
    if shared !== nothing && !isempty(shared)
        _search_functional_group(shared[1], tag) !== nothing && return true
    end

    return haskey(fv.d0, tag)
end

function Base.getindex(fv::FrameView, tag::Tuple{UInt16,UInt16})
    tag == (0x7fe0, 0x0010) && return fv.pixel_slice

    pf = haskey(fv.d0, (0x5200, 0x9230)) ? fv.d0[(0x5200, 0x9230)] : nothing
    if pf !== nothing && fv.frame_idx <= length(pf)
        v = _search_functional_group(pf[fv.frame_idx], tag)
        v !== nothing && return v
    end

    shared = haskey(fv.d0, (0x5200, 0x9229)) ? fv.d0[(0x5200, 0x9229)] : nothing
    if shared !== nothing && !isempty(shared)
        v = _search_functional_group(shared[1], tag)
        v !== nothing && return v
    end

    return fv.d0[tag]
end

"""
    extract_multiframe_pixel_slices(d0, rows, cols, n_frames)

Extracts the flat PixelData block of an Enhanced PET MultiFrame and
splits it into `n_frames` 2D matrices (`rows` x `cols`)—one per frame—
in the order in which they appear in the file.
"""
function extract_multiframe_pixel_slices(d0, rows::Int, cols::Int, n_frames::Int)
    raw = d0[(0x7fe0, 0x0010)]
    vol = reshape(Float64.(raw), rows, cols, n_frames)
    return [vol[:, :, i] for i in 1:n_frames]
end

const _UCUM_TO_INTERNAL_UNITS = Dict(
    "bq/ml" => "BQML",
    "g/ml" => "GML",
    "cm2/ml" => "CM2ML",
    "cnts" => "CNTS",
    "cps" => "CPS",
    "propcnts" => "PROPCNTS",
)

function get_units_from_rwv(d0, frame_idx::Int=1)
    groups = Any[]
    pf = get_tag(d0, (0x5200, 0x9230))
    if pf !== nothing && frame_idx <= length(pf)
        push!(groups, pf[frame_idx])
    end
    shared = get_tag(d0, (0x5200, 0x9229))
    if shared !== nothing && !isempty(shared)
        push!(groups, shared[1])
    end

    for g in groups
        rwv = get_tag(g, (0x0040, 0x9096))
        (rwv === nothing || isempty(rwv)) && continue
        entry = rwv[1]
        code_seq = get_tag(entry, (0x0040, 0x08ea))
        (code_seq === nothing || isempty(code_seq)) && continue
        code_val = sanitize(get_tag(code_seq[1], (0x0008, 0x0100)))
        base_unit = split(code_val, '{')[1]
        key = lowercase(replace(base_unit, " " => ""))
        haskey(_UCUM_TO_INTERNAL_UNITS, key) && return _UCUM_TO_INTERNAL_UNITS[key]
    end
    return ""
end