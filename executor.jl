include("src/Radiomics.jl")

using DICOM, .Radiomics, NIfTI, DataFrames

dro_root = "/home/aldo/Radiomicsaldo/DRO"

dro_dirs = sort(filter(d -> isdir(joinpath(dro_root, d)) && startswith(d, "DRO_"), readdir(dro_root)))

results = DataFrame(
    dro       = String[],
    SUVmin    = Float64[],
    SUVmed    = Float64[],
    SUVmax    = Float64[],
)

for dro_name in dro_dirs
    dro_path = joinpath(dro_root, dro_name)
    pt_path   = joinpath(dro_path, "PT")
    mask_path = joinpath(dro_path, "mask", "DRO_mask.nii.gz")

    if !isdir(pt_path) || !isfile(mask_path)
        @warn "Skipped $dro_name: PT or mask missing"
        continue
    end

    try
        dcms    = dcmdir_parse(pt_path)
        suv_vol = Radiomics.normalize_pet_and_extract_features(dcms)
        mask    = niread(mask_path)
        spacing = [mask.header.pixdim[2], mask.header.pixdim[3], mask.header.pixdim[4]]

        features = Radiomics.extract_radiomic_features(
            suv_vol, mask.raw, spacing;
            features        = [:first_order],
            keep_largest_only = true,
            sample_rate     = 1.0
        )

        push!(results, (
            dro_name,
            features["firstorder_minimum"],
            features["firstorder_median"],
            features["firstorder_maximum"],
        ))

        println("[$dro_name]  SUVmin=$(features["firstorder_minimum"])  " *
                "SUVmed=$(features["firstorder_median"])  " *
                "SUVmax=$(features["firstorder_maximum"])")

    catch e
        @warn "Errore in $dro_name: $e"
    end
end
