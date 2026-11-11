using Test
#using NIfTI
using Radiomics
using Downloads
using CUDA
using JSON3

# --------------- IBSI Validation Test Suite ---------------
"""
    IBSI Validation Test Suite

    This module contains automated tests to validate the `extract_radiomic_features` 
    function against the Image Biomarker Standardisation Initiative (IBSI) guidelines. 
    It extracts features from a standardized digital phantom and asserts that the 
    computed values strictly match the official IBSI reference values, ensuring the 
    mathematical correctness and compliance of the extraction pipeline.

    Reference Data IBSI phase 1:
    The benchmark values used for validation in this suite are derived from the 
    official IBSI Phase 1 standard. 
    The reference manuals and data sets can be accessed at:
    - Reference Data Sets Guide: https://ibsi.readthedocs.io/en/latest/05_Reference_data_sets.html
    - Official Website: https://theibsi.github.io/
    - Data Repository: https://github.com/theibsi/data_sets

"""

const PHANTOM_DIR = "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_1_digital_phantom/nifti/image/phantom.nii.gz"
const MASK_DIR = "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_1_digital_phantom/nifti/mask/mask.nii.gz"

"""
    Reference Data IBSI phase 2:
    The benchmark values used for validation in this suite are derived from the 
    official IBSI 2 (convolutional filters): digital phantoms and consensus reference response maps. 
    The reference manuals and data sets can be accessed at:
    - Data Repository: https://github.com/theibsi/ibsi_2_reference_data

"""

const IBSI2_PHANTOM_URL(name) = "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_2_digital_phantom/nifti/$name/image/$name.nii.gz"
const IBSI2_CRM_URL(id) = "https://raw.githubusercontent.com/theibsi/ibsi_2_reference_data/master/reference_response_maps/$id-ValidCRM.nii"

# All filter configurations with a consensus response map (Tab. 6.1, sez. 7.1;
# configurations 9 and 10 have no consensus)
const IBSI2_PHANTOMS = ["impulse", "checkerboard", "sphere"]
const IBSI2_CRM_IDS = [
    "1_a_1", "1_a_2", "1_a_3", "1_a_4", "1_b_1",            # mean
    "2_a", "2_b", "2_c",                                    # LoG
    "3_a_1", "3_a_2", "3_a_3", "3_b_1", "3_b_2", "3_b_3",
    "3_c_1", "3_c_2", "3_c_3",                              # Laws
    "4_a_1", "4_a_2", "4_b_1", "4_b_2",                     # Gabor
    "5_a_1", "5_a_2",                                       # Daubechies 2
    "6_a_1", "6_a_2",                                       # Coiflet 1
    "7_a_1", "7_a_2",                                       # Haar
    "8_a_1", "8_a_2", "8_a_3",                              # Simoncelli
]

"""
    ibsi_test(actual::Real, reference::Real, tolerance::Real) -> Bool

    Validates a radiomic feature against the IBSI reference value.

    Computes the absolute difference between `actual` and `reference`, rounded down
    to an adaptive number of decimal places based on the magnitude of `reference`:
    - If `reference` is large (e.g. 1000), the difference is rounded to fewer decimals.
    - If `reference` is small (e.g. 0.001), the difference is rounded to more decimals.
    - If `reference` is 0, the difference is rounded to 0 decimal places.

    # Arguments
    - `actual::Real`: The computed radiomic feature value (your result).
    - `reference::Real`: The IBSI reference value.
    - `tolerance::Real`: The IBSI tolerance value.

    # Returns
    - `true` if the rounded difference is within `tolerance` (match).
    - `false` otherwise (no match).
"""
function ibsi_test(actual::Real, reference::Real, tolerance::Real)
    if reference == 0
        diff = floor(abs(actual - reference))
    else
        decimals = 3 - (1 + floor(Int, log10(abs(reference))))
        factor = 10.0^decimals
        diff = floor(abs(actual - reference) * factor) / factor
    end
    return diff <= tolerance
end

# --------------- PET SUV Validation Test Suite --------------- 
"""
    PET SUV Validation Test Suite

    This module contains resources and validation tests to verify the correct 
    computation of standardized uptake values (SUVs) from DICOM PET images according to 
    the project's SUV computation guidelines. It uses synthetic Digital Reference Objects 
    (DROs) with known reference values to test the correctness of SUV calculation implementations 
    and a validation dataset comprising real-world PET-DICOM series and corresponding regions 
    of interest (ROIs) stored as NIfTI binary masks to assess consistency across different implementations.

    Reference Data:

    The reference materials and datasets used for SUV computation and validation are provided by the project and include:

    - SUV Computation Manual: https://oncoray.github.io/suv_computation/suv.html
    - Official Website: https://theibsi.github.io/
    - Digital Reference Objects (DROs) Repository: https://github.com/oncoray/suv_computation/tree/main/DRO
    
"""
# The hash of the reference DRO dataset.
const DRO_REF = "db3be81847f40a8484921a069ec99a39d0855aba"

# Tolerance for SUV nominal value comparison
const SUV_NOMINAL_RTOL = 0.02

# Error messages for DRO datasets
const DRO_ERROR_MSG = Dict(
    "DRO_error_2_0" => "Patient's Weight absent or not positive",
    "DRO_error_2_1" => "Patient's Weight absent or not positive",
    "DRO_error_2_2" => "error-invalid-sex",
    "DRO_error_2_3" => "error-invalid-sex",
    "DRO_error_2_4" => "error-no-height",
    "DRO_error_2_5" => "error-no-height",
    "DRO_error_2_6" => "error-CNTS-no-factor",
    "DRO_error_2_7" => "error-unit-PROPCNTS",
    "DRO_error_3_0" => "error-no-dose",
    "DRO_error_3_1" => "error-no-dose",
    "DRO_error_3_2" => "error-tref",
    "DRO_error_4_0" => "error-no-tadm",
    "DRO_error_4_1" => "error-tadm-not-date-qualified",
    "DRO_error_4_2" => "error-tadm-not-date-qualified",
    "DRO_error_5_0" => "error-no-T½",
)

"""
    download_dro_all(dest; ref=DRO_REF)

    Downloads the reference Digital Reference Objects (DROs) for SUV computation.

    This function downloads the reference DRO dataset from the official repository
    and saves it to the specified destination directory.

    # Arguments
    - `dest::AbstractString`: The destination directory where the DROs will be downloaded.
    - `ref::AbstractString`: The hash of the reference DRO dataset (default: DRO_REF).

    # Returns
    - `true` if the download was successful.
"""
function download_dro_all(dest; ref=DRO_REF)
    io = IOBuffer()
    Downloads.download("https://api.github.com/repos/oncoray/suv_computation/git/trees/$ref?recursive=1", io)
    tree = JSON3.read(String(take!(io)))
    tree.truncated && error("GitHub tree truncated")
    files = [t.path for t in tree.tree
             if t.type == "blob" && startswith(t.path, "DRO/data/") && !occursin("/RS/", t.path)]
    asyncmap(files; ntasks=8) do p
        out = joinpath(dest, relpath(p, "DRO/data"))
        mkpath(dirname(out))
        Downloads.download("https://raw.githubusercontent.com/oncoray/suv_computation/$ref/$p", out)
    end
end

# --------------- Managment of GPU --------------- 
"""
    gpu_present() -> Bool

    Check if an NVIDIA GPU is present in the system using `nvidia-smi`
"""
function gpu_present()
    try
        io = IOBuffer()
        return success(pipeline(`nvidia-smi`, stdout=io, stderr=io))
    catch
        return false
    end
end

const GPU_PRESENT = gpu_present()

if !GPU_PRESENT
    @info "No GPU detected in the system: GPU tests will be skipped."
else
    @info "GPU detected in the system: GPU tests will be done"
end

#Control of CUDA
is_gpu_available = GPU_PRESENT && CUDA.functional()

# --------------- Execution of tests --------------- 
mktempdir() do tmpdir
    phantom_path = joinpath(tmpdir, "phantom.nii.gz")
    mask_path = joinpath(tmpdir, "mask.nii.gz")
    dro_dir = joinpath(tmpdir, "DRO")
    @info "=== Start Download ===="

    @info "[STEP 1] Downloading Phantom and mask..."
    Downloads.download(PHANTOM_DIR, phantom_path)
    Downloads.download(MASK_DIR, mask_path)

    @info "[STEP 2] Downloading DROs datasets..."
    download_dro_all(dro_dir)

    @info "[STEP 3] Downloading IBSI 2 phantoms and reference response maps..."
    ibsi2_dir = joinpath(tmpdir, "IBSI2")
    mkpath(ibsi2_dir)
    for name in IBSI2_PHANTOMS
        Downloads.download(IBSI2_PHANTOM_URL(name), joinpath(ibsi2_dir, "$name.nii.gz"))
    end
    for id in IBSI2_CRM_IDS
        Downloads.download(IBSI2_CRM_URL(id), joinpath(ibsi2_dir, "$id.nii"))
    end

    @info "Download complete."

    # Managment of DROs downloaded DROs
    names = sort(filter(n -> startswith(n, "DRO_") && isdir(joinpath(dro_dir, n)), readdir(dro_dir)))

    dro_features(n) = Radiomics.extract_radiomic_features(joinpath(dro_dir, n, "PT"), joinpath(dro_dir, n, "mask", "DRO_mask.nii.gz"); suv_ibsi_normalization=true, features=[:first_order], verbose=false)

    # Test on CPU
    @info "Executing tests for CPU..."

    @info "Executing tests for 3D Features on the CPU..."

    @testset "Radiomics 3D test" begin

        radiomic_features = Radiomics.extract_radiomic_features(phantom_path, mask_path; verbose=false, n_bins=6, keep_largest_only=false)

        # ---- First Order ---- (19 features)
        @test ibsi_test(radiomic_features["firstorder_entropy"], 1.27, 0)
        @test ibsi_test(radiomic_features["firstorder_energy"], 567, 0)
        @test ibsi_test(radiomic_features["firstorder_median"], 1, 0)
        @test ibsi_test(radiomic_features["firstorder_range"], 5, 0)
        @test ibsi_test(radiomic_features["firstorder_maximum"], 6, 0)
        @test ibsi_test(radiomic_features["firstorder_robust_mean_absolute_deviation"], 1.11, 0)
        @test ibsi_test(radiomic_features["firstorder_percentile10"], 1, 0)
        @test ibsi_test(radiomic_features["firstorder_percentile90"], 4, 0)
        @test ibsi_test(radiomic_features["firstorder_minimum"], 1, 0)
        @test ibsi_test(radiomic_features["firstorder_mean"], 2.15, 0)
        @test ibsi_test(radiomic_features["firstorder_uniformity"], 0.512, 0)
        @test ibsi_test(radiomic_features["firstorder_variance"], 3.05, 0)
        @test ibsi_test(radiomic_features["firstorder_interquartile_range"], 3, 0)
        @test ibsi_test(radiomic_features["firstorder_root_mean_squared"], 2.77, 0)
        @test ibsi_test(radiomic_features["firstorder_mean_absolute_deviation"], 1.55, 0)
        @test ibsi_test(radiomic_features["firstorder_skewness"], 1.08, 0)
        @test ibsi_test(radiomic_features["firstorder_total_energy"], 4536, 0)
        @test ibsi_test(radiomic_features["firstorder_kurtosis"], 2.6453795193121667, 0)
        @test ibsi_test(radiomic_features["firstorder_standard_deviation"], 1.7570401047374198, 0)

        # ---- GLCM ---- (24 features)
        @test ibsi_test(radiomic_features["glcm_Autocorrelation"], 5.06, 0)
        @test ibsi_test(radiomic_features["glcm_ClusterProminence"], 145, 0)
        @test ibsi_test(radiomic_features["glcm_ClusterShade"], 16.6, 0)
        @test ibsi_test(radiomic_features["glcm_ClusterTendency"], 7.07, 0)
        @test ibsi_test(radiomic_features["glcm_Contrast"], 5.32, 0)
        @test ibsi_test(radiomic_features["glcm_Correlation"], 0.157, 0)
        @test ibsi_test(radiomic_features["glcm_DifferenceAverage"], 1.43, 0)
        @test ibsi_test(radiomic_features["glcm_DifferenceEntropy"], 1.56, 0)
        @test ibsi_test(radiomic_features["glcm_DifferenceVariance"], 3.06, 0)
        @test ibsi_test(radiomic_features["glcm_Id"], 0.677, 0)
        @test ibsi_test(radiomic_features["glcm_Idm"], 0.618, 0)
        @test ibsi_test(radiomic_features["glcm_Idmn"], 0.898, 0)
        @test ibsi_test(radiomic_features["glcm_Idn"], 0.851, 0)
        @test ibsi_test(radiomic_features["glcm_Imc1"], -0.157, 0)
        @test ibsi_test(radiomic_features["glcm_Imc2"], 0.52, 0)
        @test ibsi_test(radiomic_features["glcm_InverseVariance"], 0.0604, 0)
        @test ibsi_test(radiomic_features["glcm_JointAverage"], 2.14, 0)
        @test ibsi_test(radiomic_features["glcm_JointEntropy"], 2.4, 0)
        @test ibsi_test(radiomic_features["glcm_MaximumProbability"], 0.503, 0)
        @test ibsi_test(radiomic_features["glcm_SumAverage"], 4.29, 0)
        @test ibsi_test(radiomic_features["glcm_SumEntropy"], 1.92, 0)
        @test ibsi_test(radiomic_features["glcm_Mcc"], 0.44, 0)
        @test ibsi_test(radiomic_features["glcm_SumSquares"], 3.09, 0)
        @test ibsi_test(radiomic_features["glcm_JointEnergy"], 0.3029, 0)

        # ---- GLDM ---- (14 features)
        @test ibsi_test(radiomic_features["gldm_DependenceEntropy"], 4.4, 0)
        @test ibsi_test(radiomic_features["gldm_DependenceNonUniformity"], 4.86, 0)
        @test ibsi_test(radiomic_features["gldm_DependenceNonUniformityNormalized"], 0.0657, 0)
        @test ibsi_test(radiomic_features["gldm_DependenceVariance"], 22.1, 0)
        @test ibsi_test(radiomic_features["gldm_GrayLevelNonUniformity"], 37.9, 0)
        @test ibsi_test(radiomic_features["gldm_GrayLevelVariance"], 3.05, 0)
        @test ibsi_test(radiomic_features["gldm_HighGrayLevelEmphasis"], 7.66, 0)
        @test ibsi_test(radiomic_features["gldm_LargeDependenceEmphasis"], 109, 0)
        @test ibsi_test(radiomic_features["gldm_LargeDependenceHighGrayLevelEmphasis"], 235, 0)
        @test ibsi_test(radiomic_features["gldm_LargeDependenceLowGrayLevelEmphasis"], 102, 0)
        @test ibsi_test(radiomic_features["gldm_LowGrayLevelEmphasis"], 0.693, 0)
        @test ibsi_test(radiomic_features["gldm_SmallDependenceEmphasis"], 0.045, 0)
        @test ibsi_test(radiomic_features["gldm_SmallDependenceHighGrayLevelEmphasis"], 0.736, 0)
        @test ibsi_test(radiomic_features["gldm_SmallDependenceLowGrayLevelEmphasis"], 0.00963, 0)

        # ---- GLRLM ---- (16 features)
        @test ibsi_test(radiomic_features["glrlm_GrayLevelNonUniformity"], 21.8, 0)
        @test ibsi_test(radiomic_features["glrlm_GrayLevelNonUniformityNormalized"], 0.43, 0)
        @test ibsi_test(radiomic_features["glrlm_GrayLevelVariance"], 3.46, 0)
        @test ibsi_test(radiomic_features["glrlm_HighGrayLevelRunEmphasis"], 9.7, 0)
        @test ibsi_test(radiomic_features["glrlm_LongRunEmphasis"], 3.06, 0)
        @test ibsi_test(radiomic_features["glrlm_LongRunHighGrayLevelEmphasis"], 17.6, 0)
        @test ibsi_test(radiomic_features["glrlm_LongRunLowGrayLevelEmphasis"], 2.39, 0)
        @test ibsi_test(radiomic_features["glrlm_LowGrayLevelRunEmphasis"], 0.603, 0)
        @test ibsi_test(radiomic_features["glrlm_RunEntropy"], 2.43, 0)
        @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformity"], 26.9, 0)
        @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformityNormalized"], 0.513, 0)
        @test ibsi_test(radiomic_features["glrlm_RunPercentage"], 0.68, 0)
        @test ibsi_test(radiomic_features["glrlm_RunVariance"], 0.574, 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunEmphasis"], 0.705, 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunHighGrayLevelEmphasis"], 8.540, 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.352, 0)

        # ---- GLSZM ---- (16 features)
        @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformity"], 1.4, 0)
        @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformityNormalized"], 0.28, 0)
        @test ibsi_test(radiomic_features["glszm_GrayLevelVariance"], 2.64, 0)
        @test ibsi_test(radiomic_features["glszm_HighGrayLevelZoneEmphasis"], 15.6, 0)
        @test ibsi_test(radiomic_features["glszm_LargeAreaEmphasis"], 550, 0)
        @test ibsi_test(radiomic_features["glszm_LargeAreaHighGrayLevelEmphasis"], 1490, 0)
        @test ibsi_test(radiomic_features["glszm_LargeAreaLowGrayLevelEmphasis"], 503, 0)
        @test ibsi_test(radiomic_features["glszm_LowGrayLevelZoneEmphasis"], 0.253, 0)
        @test ibsi_test(radiomic_features["glszm_SizeZoneNonUniformity"], 1, 0)
        @test ibsi_test(radiomic_features["glszm_SizeZoneNonUniformityNormalized"], 0.2, 0)
        @test ibsi_test(radiomic_features["glszm_SmallAreaEmphasis"], 0.255, 0)
        @test ibsi_test(radiomic_features["glszm_SmallAreaHighGrayLevelEmphasis"], 2.76, 0)
        @test ibsi_test(radiomic_features["glszm_SmallAreaLowGrayLevelEmphasis"], 0.0256, 0)
        @test ibsi_test(radiomic_features["glszm_ZoneEntropy"], 2.32, 0)
        @test ibsi_test(radiomic_features["glszm_ZonePercentage"], 0.0676, 0)
        @test ibsi_test(radiomic_features["glszm_ZoneVariance"], 331, 0)

        # ---- NGTDM ---- (5 features)
        @test ibsi_test(radiomic_features["ngtdm_Busyness"], 6.54, 0)
        @test ibsi_test(radiomic_features["ngtdm_Coarseness"], 0.0296, 0)
        @test ibsi_test(radiomic_features["ngtdm_Complexity"], 13.5, 0)
        @test ibsi_test(radiomic_features["ngtdm_Contrast"], 0.584, 0)
        @test ibsi_test(radiomic_features["ngtdm_Strength"], 0.763, 0)

        #--- Shape3D --- (12 features)
        @test ibsi_test(radiomic_features["shape3d_surface_area"], 388, 3)
        @test ibsi_test(radiomic_features["shape3d_mesh_volume"], 556, 4)
        @test ibsi_test(radiomic_features["shape3d_surface_volume_ratio"], 0.698, 0.004)
        @test ibsi_test(radiomic_features["shape3d_sphericity"], 0.843, 0.005)
        @test ibsi_test(radiomic_features["shape3d_major_axis_length"], 11.4, 0.1)
        @test ibsi_test(radiomic_features["shape3d_maximum_3d_diameter"], 13.1, 0.1)
        @test ibsi_test(radiomic_features["shape3d_minor_axis_length"], 9.31, 0.06)
        @test ibsi_test(radiomic_features["shape3d_least_axis_length"], 8.54, 0.05)
        @test ibsi_test(radiomic_features["shape3d_elongation"], 0.816, 0.005)
        @test ibsi_test(radiomic_features["shape3d_flatness"], 0.749, 0.005)
        @test ibsi_test(radiomic_features["shape3d_voxel_volume"], 592, 4)
        @test ibsi_test(radiomic_features["shape3d_number_of_islands"], 1, 0)

    end

    @info "Executing tests for 2D Features on the CPU..."

    @testset "Radiomics 2D test" begin

        radiomic_2D_features = Radiomics.extract_radiomic_features(phantom_path, mask_path; verbose=false, n_bins=6, keep_largest_only=false, slices_2d=[(3, 3)])

        # ---- First Order ---- (19 features)
        @test ibsi_test(radiomic_2D_features["firstorder_entropy"], 0.8343470230852528, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_energy"], 82.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_median"], 1.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_range"], 5.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_maximum"], 6, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_robust_mean_absolute_deviation"], 0.65625, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_percentile10"], 1.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_percentile90"], 4.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_minimum"], 1.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_mean"], 1.6470588235294117, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_uniformity"], 0.6955017301038062, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_variance"], 2.110726643598616, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_interquartile_range"], 0.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_root_mean_squared"], 2.1962534944228786, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_mean_absolute_deviation"], 1.065743944636678, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_skewness"], 2.0091745503701466, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_total_energy"], 656.0, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_kurtosis"], 5.5822359580757865, 0)
        @test ibsi_test(radiomic_2D_features["firstorder_standard_deviation"], 1.4975470138942315, 0)

        # ---- GLCM ---- (24 features)
        @test ibsi_test(radiomic_2D_features["glcm_Autocorrelation"], 2.369002525252525, 0)
        @test ibsi_test(radiomic_2D_features["glcm_ClusterProminence"], 29.379996965839368, 0)
        @test ibsi_test(radiomic_2D_features["glcm_ClusterShade"], 6.349522162635246, 0)
        @test ibsi_test(radiomic_2D_features["glcm_ClusterTendency"], 3.379222703678196, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Contrast"], 4.176452020202021, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Correlation"], -0.11273883997997489, 0)
        @test ibsi_test(radiomic_2D_features["glcm_DifferenceAverage"], 1.0565025252525253, 0)
        @test ibsi_test(radiomic_2D_features["glcm_DifferenceEntropy"], 1.0620199761782945, 0)
        @test ibsi_test(radiomic_2D_features["glcm_DifferenceVariance"], 2.9625560370115296, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Id"], 0.7778303872053872, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Idm"], 0.7372693278943279, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Idmn"], 0.9218630981950654, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Idn"], 0.8927819865319866, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Imc1"], -0.09725061597992121, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Imc2"], 0.349416066931148, 0)
        @test ibsi_test(radiomic_2D_features["glcm_InverseVariance"], 0.02451038159371493, 0)
        @test ibsi_test(radiomic_2D_features["glcm_JointAverage"], 1.5907512626262625, 0)
        @test ibsi_test(radiomic_2D_features["glcm_JointEntropy"], 1.441418792736606, 0)
        @test ibsi_test(radiomic_2D_features["glcm_MaximumProbability"], 0.6941287878787878, 0)
        @test ibsi_test(radiomic_2D_features["glcm_SumAverage"], 3.181502525252525, 0)
        @test ibsi_test(radiomic_2D_features["glcm_SumEntropy"], 1.156380913948728, 0)
        @test ibsi_test(radiomic_2D_features["glcm_Mcc"], 0.2881064367074744, 0)
        @test ibsi_test(radiomic_2D_features["glcm_SumSquares"], 1.8889186809700542, 0)
        @test ibsi_test(radiomic_2D_features["glcm_JointEnergy"], 0.5149241387996123, 0)

        # ---- GLDM ---- (14 features)
        @test ibsi_test(radiomic_2D_features["gldm_DependenceEntropy"], 2.60985016602894, 0)
        @test ibsi_test(radiomic_2D_features["gldm_DependenceNonUniformity"], 3.1176470588235294, 0)
        @test ibsi_test(radiomic_2D_features["gldm_DependenceNonUniformityNormalized"], 0.18339100346020762, 0)
        @test ibsi_test(radiomic_2D_features["gldm_DependenceVariance"], 2.8304498269896197, 0)
        @test ibsi_test(radiomic_2D_features["gldm_GrayLevelNonUniformity"], 11.823529411764707, 0)
        @test ibsi_test(radiomic_2D_features["gldm_GrayLevelVariance"], 2.110726643598616, 0)
        @test ibsi_test(radiomic_2D_features["gldm_HighGrayLevelEmphasis"], 4.823529411764706, 0)
        @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceEmphasis"], 22.294117647058822, 0)
        @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceHighGrayLevelEmphasis"], 31.41176470588235, 0)
        @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceLowGrayLevelEmphasis"], 21.795751633986928, 0)
        @test ibsi_test(radiomic_2D_features["gldm_LowGrayLevelEmphasis"], 0.8325163398692811, 0)
        @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceEmphasis"], 0.127515339469121, 0)
        @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceHighGrayLevelEmphasis"], 2.627515339469121, 0)
        @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceLowGrayLevelEmphasis"], 0.04275226757369615, 0)

        # ---- GLRLM ---- (16 features)
        @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelNonUniformity"], 5.484848484848485, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelNonUniformityNormalized"], 0.55749253902663, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelVariance"], 3.096630366161616, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_HighGrayLevelRunEmphasis"], 7.4251893939393945, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_LongRunEmphasis"], 4.246212121212121, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_LongRunHighGrayLevelEmphasis"], 12.077651515151516, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_LongRunLowGrayLevelEmphasis"], 3.885640256734007, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_LowGrayLevelRunEmphasis"], 0.7273187605218856, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_RunEntropy"], 2.0820756102292672, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_RunLengthNonUniformity"], 4.496212121212121, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_RunLengthNonUniformityNormalized"], 0.4567263544536272, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_RunPercentage"], 0.573529411764706, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_RunVariance"], 0.8895560720844812, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_ShortRunEmphasis"], 0.5864049347643098, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_ShortRunHighGrayLevelEmphasis"], 6.660031828703703, 0)
        @test ibsi_test(radiomic_2D_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.33569635153619526, 0)

        # ---- GLSZM ---- (16 features)
        @test ibsi_test(radiomic_2D_features["glszm_GrayLevelNonUniformity"], 1.0, 0)
        @test ibsi_test(radiomic_2D_features["glszm_GrayLevelNonUniformityNormalized"], 0.3333333333333333, 0)
        @test ibsi_test(radiomic_2D_features["glszm_GrayLevelVariance"], 4.222222222222222, 0)
        @test ibsi_test(radiomic_2D_features["glszm_HighGrayLevelZoneEmphasis"], 17.666666666666668, 0)
        @test ibsi_test(radiomic_2D_features["glszm_LargeAreaEmphasis"], 67.0, 0)
        @test ibsi_test(radiomic_2D_features["glszm_LargeAreaHighGrayLevelEmphasis"], 98.66666666666667, 0)
        @test ibsi_test(radiomic_2D_features["glszm_LargeAreaLowGrayLevelEmphasis"], 65.42592592592592, 0)
        @test ibsi_test(radiomic_2D_features["glszm_LowGrayLevelZoneEmphasis"], 0.3634259259259259, 0)
        @test ibsi_test(radiomic_2D_features["glszm_SizeZoneNonUniformity"], 1.0, 0)
        @test ibsi_test(radiomic_2D_features["glszm_SizeZoneNonUniformityNormalized"], 0.3333333333333333, 0)
        @test ibsi_test(radiomic_2D_features["glszm_SmallAreaEmphasis"], 0.4183673469387755, 0)
        @test ibsi_test(radiomic_2D_features["glszm_SmallAreaHighGrayLevelEmphasis"], 13.335034013605442, 0)
        @test ibsi_test(radiomic_2D_features["glszm_SmallAreaLowGrayLevelEmphasis"], 0.016168272864701436, 0)
        @test ibsi_test(radiomic_2D_features["glszm_ZoneEntropy"], 1.5849625007211552, 0)
        @test ibsi_test(radiomic_2D_features["glszm_ZonePercentage"], 0.17647058823529413, 0)
        @test ibsi_test(radiomic_2D_features["glszm_ZoneVariance"], 34.888888888888886, 0)

        # ---- NGTDM ---- (5 features)
        @test ibsi_test(radiomic_2D_features["ngtdm_Busyness"], 3.940625, 0)
        @test ibsi_test(radiomic_2D_features["ngtdm_Coarseness"], 0.13481363996827916, 0)
        @test ibsi_test(radiomic_2D_features["ngtdm_Complexity"], 8.37328431372549, 0)
        @test ibsi_test(radiomic_2D_features["ngtdm_Contrast"], 0.7298857259166641, 0)
        @test ibsi_test(radiomic_2D_features["ngtdm_Strength"], 3.5422771781859765, 0)

        #--- Shape2D --- 
        @test ibsi_test(radiomic_2D_features["shape2d_elongation"], 0.7143974655883708, 0.005)
        @test ibsi_test(radiomic_2D_features["shape2d_major_axis_length"], 11.63955229821705, 0.1)
        @test ibsi_test(radiomic_2D_features["shape2d_maximum_diameter"], 11.661903789690601, 0.1)
        @test ibsi_test(radiomic_2D_features["shape2d_mesh_surface"], 68.0, 4)
        @test ibsi_test(radiomic_2D_features["shape2d_minor_axis_length"], 8.315266662429558, 0.06)
        @test ibsi_test(radiomic_2D_features["shape2d_perimeter"], 38.14213562373095, 0)
        @test ibsi_test(radiomic_2D_features["shape2d_perimeter_surface_ratio"], 0.560913759172514, 0.004)
        @test ibsi_test(radiomic_2D_features["shape2d_pixel_surface"], 68.0, 0)
        @test ibsi_test(radiomic_2D_features["shape2d_sphericity"], 0.7663980345420807, 0.005)
        @test ibsi_test(radiomic_2D_features["shape2d_number_of_islands"], 1, 0)

    end

    @info "Executing tests for DRO dataset features on the CPU..."

    @testset "Radiomics DROs set" begin
        for n in names
            @testset "$n" begin
                if startswith(n, "DRO_error")
                    @test_throws DRO_ERROR_MSG[n] dro_features(n)
                else
                    radiomic_features = dro_features(n)

                    # Nominal SUV values for each of ROIs: (SUVmin, SUVmediana, SUVmax)
                    @test isapprox(radiomic_features["firstorder_minimum"], 0.20, rtol=SUV_NOMINAL_RTOL)
                    @test isapprox(radiomic_features["firstorder_median"], 1.0, rtol=SUV_NOMINAL_RTOL)
                    @test isapprox(radiomic_features["firstorder_maximum"], 4.0, rtol=SUV_NOMINAL_RTOL)
                end
            end
        end

    end

    @info "Executing tests for Haar filter on the CPU..."

    @testset "Haar Filter" begin
        vol = rand(16, 16, 16); sp1 = [1.0, 1.0, 1.0]

        # Without rotation invariance: all 8 subbands, same size as the image, no log messages
        r = @test_logs Radiomics.haar_wavelet_filter(vol, sp1; rotation_invariant=false)
        @test length(r) == 8
        @test all(size(v) == size(vol) for v in values(r))

        # Rotation invariance + "all": only the distinct subbands, with @info
        r = @test_logs (:info, r"returning only") Radiomics.haar_wavelet_filter(vol, sp1)
        @test sort(collect(keys(r))) == sort(["wavelet-LLL", "wavelet-LLH", "wavelet-LHH", "wavelet-HHH"])
        r2d = @test_logs (:info, r"returning only") Radiomics.haar_wavelet_filter(rand(16, 16), [1.0, 1.0])
        @test sort(collect(keys(r2d))) == sort(["wavelet-LL", "wavelet-LH", "wavelet-HH"])

        # Equivalent subbands are identical after pooling; requesting them gives one @warn each
        r = @test_logs (:warn, r"requested LLH") (:warn, r"requested LHL") (:warn, r"requested HLL") Radiomics.haar_wavelet_filter(vol, sp1; subbands=["LLH", "LHL", "HLL"])
        @test r["wavelet-LLH"] ≈ r["wavelet-LHL"] ≈ r["wavelet-HLL"]
        # All-L and all-H subbands have no equivalents: no warning
        @test_logs Radiomics.haar_wavelet_filter(vol, sp1; subbands=["LLL", "HHH"])

        # Constant image, mirror padding: LLL = c·(√2)^3, details = 0
        c = Radiomics.haar_wavelet_filter(fill(5.0, 8, 8, 8), sp1; subbands=["LLL", "HHH"])
        @test all(c["wavelet-LLL"] .≈ 5 * sqrt(2)^3)
        @test maximum(abs, c["wavelet-HHH"]) < 1e-12

        # Levels: details for every level, approximation only for the last one
        r = Radiomics.haar_wavelet_filter(vol, sp1; level=2, subbands=["LLL", "HHH"])
        @test sort(collect(keys(r))) == ["wavelet-HHH", "wavelet2-HHH", "wavelet2-LLL"]

        # Invalid arguments
        @test_throws ErrorException Radiomics.haar_wavelet_filter(vol, sp1; level=0)
        @test_throws ErrorException Radiomics.haar_wavelet_filter(vol, sp1; padding="foo")
        @test_throws ErrorException Radiomics.haar_wavelet_filter(vol, sp1; pooling="foo")
        @test_throws ErrorException Radiomics.haar_wavelet_filter(vol, sp1; subbands=["LLX"])
        @test_throws MethodError Radiomics.haar_wavelet_filter(vol, sp1; sigma=1.0)

        # Integration with extract_radiomic_features
        hf = Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="haar", filter_params=(level=1, subbands=["LLL"]), features=[:first_order],
            n_bins=6, keep_largest_only=false)
        @test haskey(hf, "firstorder_mean")
        @test_throws MethodError Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="haar", filter_params=(sigma=1.0,), features=[:first_order])
        @test Radiomics.get_filter("haar") === Radiomics.haar_wavelet_filter

    end

    @info "Executing tests for Gabor filter on the CPU..."

    @testset "Gabor Filter" begin
        # gabor_filter returns a one-entry Dict (response => map): extract the map
        gabor_map(args...; kwargs...) = only(values(Radiomics.gabor_filter(args...; kwargs...)))

        # Output format: one entry keyed by the response type
        imp = zeros(33, 33); imp[17, 17] = 1.0
        @test collect(keys(Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=4.0, lambda=5.0))) == ["modulus"]
        @test collect(keys(Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=4.0, lambda=5.0, response="real"))) == ["real"]

        # Impulse response: the modulus is the Gaussian envelope of the kernel
        h = gabor_map(imp, [1.0, 1.0]; sigma=4.0, lambda=5.0, gamma=1.0, theta=0.0)
        @test size(h) == size(imp)
        @test h[17, 17] ≈ 1.0
        @test h[17+3, 17] ≈ exp(-3^2 / (2 * 4.0^2))
        @test h[17, 17+3] ≈ exp(-(1.0^2 * 3^2) / (2 * 4.0^2))

        # Orientation: theta=0 has the oscillation (and gamma-free axis) along dim 1
        h_g = gabor_map(imp, [1.0, 1.0]; sigma=4.0, lambda=5.0, gamma=2.0, theta=0.0)
        @test h_g[17+3, 17] ≈ exp(-3^2 / (2 * 4.0^2))
        @test h_g[17, 17+3] ≈ exp(-(2.0^2 * 3^2) / (2 * 4.0^2))

        # Modulus does not depend on theta -> theta + pi
        img = rand(20, 20)
        @test gabor_map(img, [1.0, 1.0]; sigma=3.0, lambda=4.0, theta=0.3) ≈
              gabor_map(img, [1.0, 1.0]; sigma=3.0, lambda=4.0, theta=0.3 + π)

        # Physical units: sigma/lambda in mm scale with spacing
        imp2 = zeros(33, 33); imp2[17, 17] = 1.0
        h2 = gabor_map(imp2, [2.0, 2.0]; sigma=8.0, lambda=10.0)
        h1 = gabor_map(imp2, [1.0, 1.0]; sigma=4.0, lambda=5.0)
        @test h1 ≈ h2

        # Pooling and 3D / orthogonal planes keep the image shape
        vol = rand(10, 12, 8)
        @test size(gabor_map(vol, [1.0, 1.0, 2.0]; sigma=2.0, lambda=3.0)) == size(vol)
        @test size(gabor_map(vol, [1.0, 1.0, 2.0]; sigma=2.0, lambda=3.0,
            rotation_invariant=true, delta_theta=π / 2, pooling="max", orthogonal_planes=true, padding="mirror")) == size(vol)

        # Invalid arguments
        @test_throws ErrorException Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=-1.0, lambda=5.0)
        @test_throws ErrorException Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=1.0, lambda=5.0, padding="foo")
        @test_throws ErrorException Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=1.0, lambda=5.0, orthogonal_planes=true)
        # Unknown keyword (e.g. a wavelet parameter) is rejected
        @test_throws MethodError Radiomics.gabor_filter(imp, [1.0, 1.0]; sigma=1.0, lambda=5.0, level=2)

        # Integration with extract_radiomic_features
        gf = Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="gabor", filter_params=(sigma=2.0, lambda=3.0, gamma=1.0), features=[:first_order],
            n_bins=6, keep_largest_only=false)
        @test haskey(gf, "firstorder_mean")
        @test_throws ErrorException Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="gabor", filter_params=(sigma=2.0, lambda=3.0), features=[:shape3d])
        @test_throws MethodError Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="gabor", filter_params=(sigma=2.0, lambda=3.0, level=2), features=[:first_order])
        @test Radiomics.get_filter("gabor") === Radiomics.gabor_filter
    end

    @info "Executing tests for LoG filter on the CPU..."

    @testset "LoG Filter" begin
        # Impulse response of an interior voxel equals the (truncated) 3D kernel; zero-sum check
        imp = zeros(41, 41, 41); imp[21, 21, 21] = 1.0
        h = only(values(Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=3.0)))
        σ = 1.5
        g0 = -(1 / σ^2) * (1 / (sqrt(2π) * σ))^3 * 3
        @test h[21, 21, 21] ≈ g0
        @test isapprox(sum(h), 0.0; atol=0.03 * abs(g0))

        # Output keys are pyradiomics-style, one image per sigma
        r = Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=[1.5, 3.0])
        @test sort(collect(keys(r))) == ["log-sigma-1-5-mm-3D", "log-sigma-3-0-mm-3D"]
        r2 = Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=3.0, dims=2)
        @test haskey(r2, "log-sigma-3-0-mm-2D")

        # Constant image with mirror padding gives zero response (kernel sums to ~0)
        c = Radiomics.log_filter(fill(5.0, 12, 12, 12), [1.0, 1.0, 1.0]; sigma=1.0, padding="mirror")
        @test maximum(abs, only(values(c))) < 0.05

        # Invalid arguments
        @test_throws ErrorException Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=-1.0)
        @test_throws ErrorException Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=1.0, padding="foo")
        @test_throws ErrorException Radiomics.log_filter(rand(8, 8), [1.0, 1.0]; sigma=1.0)
        # Unknown keyword (e.g. a wavelet parameter) is rejected
        @test_throws MethodError Radiomics.log_filter(imp, [2.0, 2.0, 2.0]; sigma=1.0, level=2)

        # Integration through the filter register
        lf = Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="log", filter_params=(sigma=[1.0, 2.0],), features=[:first_order],
            n_bins=6, keep_largest_only=false)
        @test sort(collect(keys(lf))) == ["log_log-sigma-1-0-mm-3D", "log_log-sigma-2-0-mm-3D"]
        @test haskey(lf["log_log-sigma-1-0-mm-3D"], "firstorder_mean")
        @test_throws MethodError Radiomics.extract_radiomic_features(phantom_path, mask_path;
            filter_type="log", filter_params=(sigma=1.0, level=2), features=[:first_order])
        @test Radiomics.get_filter("log") === Radiomics.log_filter
    end

    @info "Executing IBSI 2 reference response map tests (Gabor, LoG, Haar) on the CPU..."

    @testset "IBSI 2 reference response maps" begin
        # Phantoms and consensus response maps are downloaded in STEP 4 into ibsi2_dir
        phantom(name) = Radiomics.read_image(joinpath(ibsi2_dir, "$name.nii.gz"))
        crm(id) = Float64.(Radiomics.read_image(joinpath(ibsi2_dir, "$id.nii")).data)

        # IBSI 2 compliance (reference manual, sec. 7.1): every voxel within 1% of the reference intensity range
        function ibsi2_compliant(h, ref)
            return size(h) == size(ref) && maximum(abs.(h .- ref)) <= 0.01 * (maximum(ref) - minimum(ref))
        end
        # Filters return a Dict of response maps: these tests produce exactly one
        one_map(d) = only(values(d))

        impulse_img = phantom("impulse")
        sp = impulse_img.spacing   # IBSI 2 digital phantoms: 64x64x64 voxels, 2 mm isotropic (from header)

        impulse = Float64.(impulse_img.data)
        sphere = Float64.(phantom("sphere").data)
        checkerboard = Float64.(phantom("checkerboard").data)

        @testset "LoG" begin
            @test ibsi2_compliant(one_map(Radiomics.log_filter(impulse, sp; sigma=3.0, padding="zero")), crm("2_a"))                    # 2.a
            @test ibsi2_compliant(one_map(Radiomics.log_filter(checkerboard, sp; sigma=5.0, padding="mirror")), crm("2_b"))             # 2.b
            @test ibsi2_compliant(one_map(Radiomics.log_filter(checkerboard, sp; sigma=5.0, padding="mirror", dims=2)), crm("2_c"))     # 2.c
        end

        @testset "Gabor" begin
            # 4.a.1: single orientation, zero padding
            @test ibsi2_compliant(one_map(Radiomics.gabor_filter(impulse, sp; sigma=10.0, lambda=4.0, gamma=0.5, theta=π / 3)), crm("4_a_1"))
            # 4.a.2: rotation invariance (average pooling) + average over orthogonal planes
            @test ibsi2_compliant(one_map(Radiomics.gabor_filter(impulse, sp; sigma=10.0, lambda=4.0, gamma=0.5,
                rotation_invariant=true, delta_theta=π / 4, orthogonal_planes=true)), crm("4_a_2"))
            # 4.b.1: single orientation, mirror padding
            @test ibsi2_compliant(one_map(Radiomics.gabor_filter(sphere, sp; sigma=20.0, lambda=8.0, gamma=2.5, theta=5π / 4, padding="mirror")), crm("4_b_1"))
            # 4.b.2: rotation invariance + orthogonal planes
            @test ibsi2_compliant(one_map(Radiomics.gabor_filter(sphere, sp; sigma=20.0, lambda=8.0, gamma=2.5, padding="mirror",
                rotation_invariant=true, delta_theta=π / 8, orthogonal_planes=true)), crm("4_b_2"))
        end

        @testset "Haar" begin
            # 7.a.1 / 7.a.2: 3D, 2nd level, mirror padding, rotation invariance (average pooling)
            haar = Radiomics.haar_wavelet_filter(checkerboard, sp; level=2, subbands=["LLL", "HHH"],
                padding="mirror", rotation_invariant=true, pooling="average")
            @test ibsi2_compliant(haar["wavelet2-LLL"], crm("7_a_1"))   # 7.a.1
            @test ibsi2_compliant(haar["wavelet2-HHH"], crm("7_a_2"))   # 7.a.2
        end

    end

    # Test on GPU
    if is_gpu_available

        println("======================================================")
        @info "Executing tests for GPU..."

        @info "Executing tests for 3D Features on the GPU..."

        @testset "Radiomics 3D test GPU" begin

            radiomic_features = Radiomics.extract_radiomic_features(phantom_path, mask_path; verbose=false, use_gpu=true, n_bins=6, keep_largest_only=false)

            # ---- First Order ---- (19 features)
            @test ibsi_test(radiomic_features["firstorder_entropy"], 1.27, 0)
            @test ibsi_test(radiomic_features["firstorder_energy"], 567, 0)
            @test ibsi_test(radiomic_features["firstorder_median"], 1, 0)
            @test ibsi_test(radiomic_features["firstorder_range"], 5, 0)
            @test ibsi_test(radiomic_features["firstorder_maximum"], 6, 0)
            @test ibsi_test(radiomic_features["firstorder_robust_mean_absolute_deviation"], 1.11, 0)
            @test ibsi_test(radiomic_features["firstorder_percentile10"], 1, 0)
            @test ibsi_test(radiomic_features["firstorder_percentile90"], 4, 0)
            @test ibsi_test(radiomic_features["firstorder_minimum"], 1, 0)
            @test ibsi_test(radiomic_features["firstorder_mean"], 2.15, 0)
            @test ibsi_test(radiomic_features["firstorder_uniformity"], 0.512, 0)
            @test ibsi_test(radiomic_features["firstorder_variance"], 3.05, 0)
            @test ibsi_test(radiomic_features["firstorder_interquartile_range"], 3, 0)
            @test ibsi_test(radiomic_features["firstorder_root_mean_squared"], 2.77, 0)
            @test ibsi_test(radiomic_features["firstorder_mean_absolute_deviation"], 1.55, 0)
            @test ibsi_test(radiomic_features["firstorder_skewness"], 1.08, 0)
            @test ibsi_test(radiomic_features["firstorder_total_energy"], 4536, 0)
            @test ibsi_test(radiomic_features["firstorder_kurtosis"], 2.6453795193121667, 0)
            @test ibsi_test(radiomic_features["firstorder_standard_deviation"], 1.7570401047374198, 0)

            # ---- GLCM ---- (24 features)
            @test ibsi_test(radiomic_features["glcm_Autocorrelation"], 5.06, 0)
            @test ibsi_test(radiomic_features["glcm_ClusterProminence"], 145, 0)
            @test ibsi_test(radiomic_features["glcm_ClusterShade"], 16.6, 0)
            @test ibsi_test(radiomic_features["glcm_ClusterTendency"], 7.07, 0)
            @test ibsi_test(radiomic_features["glcm_Contrast"], 5.32, 0)
            @test ibsi_test(radiomic_features["glcm_Correlation"], 0.157, 0)
            @test ibsi_test(radiomic_features["glcm_DifferenceAverage"], 1.43, 0)
            @test ibsi_test(radiomic_features["glcm_DifferenceEntropy"], 1.56, 0)
            @test ibsi_test(radiomic_features["glcm_DifferenceVariance"], 3.06, 0)
            @test ibsi_test(radiomic_features["glcm_Id"], 0.677, 0)
            @test ibsi_test(radiomic_features["glcm_Idm"], 0.618, 0)
            @test ibsi_test(radiomic_features["glcm_Idmn"], 0.898, 0)
            @test ibsi_test(radiomic_features["glcm_Idn"], 0.851, 0)
            @test ibsi_test(radiomic_features["glcm_Imc1"], -0.157, 0)
            @test ibsi_test(radiomic_features["glcm_Imc2"], 0.52, 0)
            @test ibsi_test(radiomic_features["glcm_InverseVariance"], 0.0604, 0)
            @test ibsi_test(radiomic_features["glcm_JointAverage"], 2.14, 0)
            @test ibsi_test(radiomic_features["glcm_JointEntropy"], 2.4, 0)
            @test ibsi_test(radiomic_features["glcm_MaximumProbability"], 0.503, 0)
            @test ibsi_test(radiomic_features["glcm_SumAverage"], 4.29, 0)
            @test ibsi_test(radiomic_features["glcm_SumEntropy"], 1.92, 0)
            @test ibsi_test(radiomic_features["glcm_Mcc"], 0.44, 0)
            @test ibsi_test(radiomic_features["glcm_SumSquares"], 3.09, 0)
            @test ibsi_test(radiomic_features["glcm_JointEnergy"], 0.3029, 0)

            # ---- GLDM ---- (14 features)
            @test ibsi_test(radiomic_features["gldm_DependenceEntropy"], 4.4, 0)
            @test ibsi_test(radiomic_features["gldm_DependenceNonUniformity"], 4.86, 0)
            @test ibsi_test(radiomic_features["gldm_DependenceNonUniformityNormalized"], 0.0657, 0)
            @test ibsi_test(radiomic_features["gldm_DependenceVariance"], 22.1, 0)
            @test ibsi_test(radiomic_features["gldm_GrayLevelNonUniformity"], 37.9, 0)
            @test ibsi_test(radiomic_features["gldm_GrayLevelVariance"], 3.05, 0)
            @test ibsi_test(radiomic_features["gldm_HighGrayLevelEmphasis"], 7.66, 0)
            @test ibsi_test(radiomic_features["gldm_LargeDependenceEmphasis"], 109, 0)
            @test ibsi_test(radiomic_features["gldm_LargeDependenceHighGrayLevelEmphasis"], 235, 0)
            @test ibsi_test(radiomic_features["gldm_LargeDependenceLowGrayLevelEmphasis"], 102, 0)
            @test ibsi_test(radiomic_features["gldm_LowGrayLevelEmphasis"], 0.693, 0)
            @test ibsi_test(radiomic_features["gldm_SmallDependenceEmphasis"], 0.045, 0)
            @test ibsi_test(radiomic_features["gldm_SmallDependenceHighGrayLevelEmphasis"], 0.736, 0)
            @test ibsi_test(radiomic_features["gldm_SmallDependenceLowGrayLevelEmphasis"], 0.00963, 0)

            # ---- GLRLM ---- (16 features)
            @test ibsi_test(radiomic_features["glrlm_GrayLevelNonUniformity"], 21.8, 0)
            @test ibsi_test(radiomic_features["glrlm_GrayLevelNonUniformityNormalized"], 0.43, 0)
            @test ibsi_test(radiomic_features["glrlm_GrayLevelVariance"], 3.46, 0)
            @test ibsi_test(radiomic_features["glrlm_HighGrayLevelRunEmphasis"], 9.7, 0)
            @test ibsi_test(radiomic_features["glrlm_LongRunEmphasis"], 3.06, 0)
            @test ibsi_test(radiomic_features["glrlm_LongRunHighGrayLevelEmphasis"], 17.6, 0)
            @test ibsi_test(radiomic_features["glrlm_LongRunLowGrayLevelEmphasis"], 2.39, 0)
            @test ibsi_test(radiomic_features["glrlm_LowGrayLevelRunEmphasis"], 0.603, 0)
            @test ibsi_test(radiomic_features["glrlm_RunEntropy"], 2.43, 0)
            @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformity"], 26.9, 0)
            @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformityNormalized"], 0.513, 0)
            @test ibsi_test(radiomic_features["glrlm_RunPercentage"], 0.68, 0)
            @test ibsi_test(radiomic_features["glrlm_RunVariance"], 0.574, 0)
            @test ibsi_test(radiomic_features["glrlm_ShortRunEmphasis"], 0.705, 0)
            @test ibsi_test(radiomic_features["glrlm_ShortRunHighGrayLevelEmphasis"], 8.540, 0)
            @test ibsi_test(radiomic_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.352, 0)

            # ---- GLSZM ---- (16 features)
            @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformity"], 1.4, 0)
            @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformityNormalized"], 0.28, 0)
            @test ibsi_test(radiomic_features["glszm_GrayLevelVariance"], 2.64, 0)
            @test ibsi_test(radiomic_features["glszm_HighGrayLevelZoneEmphasis"], 15.6, 0)
            @test ibsi_test(radiomic_features["glszm_LargeAreaEmphasis"], 550, 0)
            @test ibsi_test(radiomic_features["glszm_LargeAreaHighGrayLevelEmphasis"], 1490, 0)
            @test ibsi_test(radiomic_features["glszm_LargeAreaLowGrayLevelEmphasis"], 503, 0)
            @test ibsi_test(radiomic_features["glszm_LowGrayLevelZoneEmphasis"], 0.253, 0)
            @test ibsi_test(radiomic_features["glszm_SizeZoneNonUniformity"], 1, 0)
            @test ibsi_test(radiomic_features["glszm_SizeZoneNonUniformityNormalized"], 0.2, 0)
            @test ibsi_test(radiomic_features["glszm_SmallAreaEmphasis"], 0.255, 0)
            @test ibsi_test(radiomic_features["glszm_SmallAreaHighGrayLevelEmphasis"], 2.76, 0)
            @test ibsi_test(radiomic_features["glszm_SmallAreaLowGrayLevelEmphasis"], 0.0256, 0)
            @test ibsi_test(radiomic_features["glszm_ZoneEntropy"], 2.32, 0)
            @test ibsi_test(radiomic_features["glszm_ZonePercentage"], 0.0676, 0)
            @test ibsi_test(radiomic_features["glszm_ZoneVariance"], 331, 0)

            # ---- NGTDM ---- (5 features)
            @test ibsi_test(radiomic_features["ngtdm_Busyness"], 6.54, 0)
            @test ibsi_test(radiomic_features["ngtdm_Coarseness"], 0.0296, 0)
            @test ibsi_test(radiomic_features["ngtdm_Complexity"], 13.5, 0)
            @test ibsi_test(radiomic_features["ngtdm_Contrast"], 0.584, 0)
            @test ibsi_test(radiomic_features["ngtdm_Strength"], 0.763, 0)

            #--- Shape3D --- (12 features)
            @test ibsi_test(radiomic_features["shape3d_surface_area"], 388, 3)
            @test ibsi_test(radiomic_features["shape3d_mesh_volume"], 556, 4)
            @test ibsi_test(radiomic_features["shape3d_surface_volume_ratio"], 0.698, 0.004)
            @test ibsi_test(radiomic_features["shape3d_sphericity"], 0.843, 0.005)
            @test ibsi_test(radiomic_features["shape3d_major_axis_length"], 11.4, 0.1)
            @test ibsi_test(radiomic_features["shape3d_maximum_2d_diameter_column"], 11.661903789690601, 0)
            @test ibsi_test(radiomic_features["shape3d_maximum_2d_diameter_row"], 10.0, 0)
            @test ibsi_test(radiomic_features["shape3d_maximum_2d_diameter_slice"], 11.661903789690601, 0)
            @test ibsi_test(radiomic_features["shape3d_maximum_3d_diameter"], 13.1, 0.1)
            @test ibsi_test(radiomic_features["shape3d_minor_axis_length"], 9.31, 0.06)
            @test ibsi_test(radiomic_features["shape3d_least_axis_length"], 8.54, 0.05)
            @test ibsi_test(radiomic_features["shape3d_elongation"], 0.816, 0.005)
            @test ibsi_test(radiomic_features["shape3d_flatness"], 0.749, 0.005)
            @test ibsi_test(radiomic_features["shape3d_voxel_volume"], 592, 4)
            @test ibsi_test(radiomic_features["shape3d_number_of_islands"], 1, 0)

        end

        @info "Executing tests for 2D Features on the GPU..."

        @testset "Radiomics 2D test GPU" begin

            radiomic_2D_features = Radiomics.extract_radiomic_features(phantom_path, mask_path; verbose=false, n_bins=6, use_gpu=true, keep_largest_only=false, slices_2d=[(3, 3)])

            # ---- First Order ---- (19 features)
            @test ibsi_test(radiomic_2D_features["firstorder_entropy"], 0.8343470230852528, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_energy"], 82.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_median"], 1.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_range"], 5.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_maximum"], 6, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_robust_mean_absolute_deviation"], 0.65625, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_percentile10"], 1.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_percentile90"], 4.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_minimum"], 1.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_mean"], 1.6470588235294117, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_uniformity"], 0.6955017301038062, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_variance"], 2.110726643598616, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_interquartile_range"], 0.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_root_mean_squared"], 2.1962534944228786, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_mean_absolute_deviation"], 1.065743944636678, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_skewness"], 2.0091745503701466, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_total_energy"], 656.0, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_kurtosis"], 5.5822359580757865, 0)
            @test ibsi_test(radiomic_2D_features["firstorder_standard_deviation"], 1.4975470138942315, 0)

            # ---- GLCM ---- (24 features)
            @test ibsi_test(radiomic_2D_features["glcm_Autocorrelation"], 2.369002525252525, 0)
            @test ibsi_test(radiomic_2D_features["glcm_ClusterProminence"], 29.379996965839368, 0)
            @test ibsi_test(radiomic_2D_features["glcm_ClusterShade"], 6.349522162635246, 0)
            @test ibsi_test(radiomic_2D_features["glcm_ClusterTendency"], 3.379222703678196, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Contrast"], 4.176452020202021, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Correlation"], -0.11273883997997489, 0)
            @test ibsi_test(radiomic_2D_features["glcm_DifferenceAverage"], 1.0565025252525253, 0)
            @test ibsi_test(radiomic_2D_features["glcm_DifferenceEntropy"], 1.0620199761782945, 0)
            @test ibsi_test(radiomic_2D_features["glcm_DifferenceVariance"], 2.9625560370115296, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Id"], 0.7778303872053872, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Idm"], 0.7372693278943279, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Idmn"], 0.9218630981950654, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Idn"], 0.8927819865319866, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Imc1"], -0.09725061597992121, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Imc2"], 0.349416066931148, 0)
            @test ibsi_test(radiomic_2D_features["glcm_InverseVariance"], 0.02451038159371493, 0)
            @test ibsi_test(radiomic_2D_features["glcm_JointAverage"], 1.5907512626262625, 0)
            @test ibsi_test(radiomic_2D_features["glcm_JointEntropy"], 1.441418792736606, 0)
            @test ibsi_test(radiomic_2D_features["glcm_MaximumProbability"], 0.6941287878787878, 0)
            @test ibsi_test(radiomic_2D_features["glcm_SumAverage"], 3.181502525252525, 0)
            @test ibsi_test(radiomic_2D_features["glcm_SumEntropy"], 1.156380913948728, 0)
            @test ibsi_test(radiomic_2D_features["glcm_Mcc"], 0.2881064367074744, 0)
            @test ibsi_test(radiomic_2D_features["glcm_SumSquares"], 1.8889186809700542, 0)
            @test ibsi_test(radiomic_2D_features["glcm_JointEnergy"], 0.5149241387996123, 0)

            # ---- GLDM ---- (14 features)
            @test ibsi_test(radiomic_2D_features["gldm_DependenceEntropy"], 2.60985016602894, 0)
            @test ibsi_test(radiomic_2D_features["gldm_DependenceNonUniformity"], 3.1176470588235294, 0)
            @test ibsi_test(radiomic_2D_features["gldm_DependenceNonUniformityNormalized"], 0.18339100346020762, 0)
            @test ibsi_test(radiomic_2D_features["gldm_DependenceVariance"], 2.8304498269896197, 0)
            @test ibsi_test(radiomic_2D_features["gldm_GrayLevelNonUniformity"], 11.823529411764707, 0)
            @test ibsi_test(radiomic_2D_features["gldm_GrayLevelVariance"], 2.110726643598616, 0)
            @test ibsi_test(radiomic_2D_features["gldm_HighGrayLevelEmphasis"], 4.823529411764706, 0)
            @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceEmphasis"], 22.294117647058822, 0)
            @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceHighGrayLevelEmphasis"], 31.41176470588235, 0)
            @test ibsi_test(radiomic_2D_features["gldm_LargeDependenceLowGrayLevelEmphasis"], 21.795751633986928, 0)
            @test ibsi_test(radiomic_2D_features["gldm_LowGrayLevelEmphasis"], 0.8325163398692811, 0)
            @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceEmphasis"], 0.127515339469121, 0)
            @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceHighGrayLevelEmphasis"], 2.627515339469121, 0)
            @test ibsi_test(radiomic_2D_features["gldm_SmallDependenceLowGrayLevelEmphasis"], 0.04275226757369615, 0)

            # ---- GLRLM ---- (16 features)
            @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelNonUniformity"], 5.484848484848485, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelNonUniformityNormalized"], 0.55749253902663, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_GrayLevelVariance"], 3.096630366161616, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_HighGrayLevelRunEmphasis"], 7.4251893939393945, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_LongRunEmphasis"], 4.246212121212121, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_LongRunHighGrayLevelEmphasis"], 12.077651515151516, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_LongRunLowGrayLevelEmphasis"], 3.885640256734007, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_LowGrayLevelRunEmphasis"], 0.7273187605218856, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_RunEntropy"], 2.0820756102292672, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_RunLengthNonUniformity"], 4.496212121212121, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_RunLengthNonUniformityNormalized"], 0.4567263544536272, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_RunPercentage"], 0.573529411764706, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_RunVariance"], 0.8895560720844812, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_ShortRunEmphasis"], 0.5864049347643098, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_ShortRunHighGrayLevelEmphasis"], 6.660031828703703, 0)
            @test ibsi_test(radiomic_2D_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.33569635153619526, 0)

            # ---- GLSZM ---- (16 features)
            @test ibsi_test(radiomic_2D_features["glszm_GrayLevelNonUniformity"], 1.0, 0)
            @test ibsi_test(radiomic_2D_features["glszm_GrayLevelNonUniformityNormalized"], 0.3333333333333333, 0)
            @test ibsi_test(radiomic_2D_features["glszm_GrayLevelVariance"], 4.222222222222222, 0)
            @test ibsi_test(radiomic_2D_features["glszm_HighGrayLevelZoneEmphasis"], 17.666666666666668, 0)
            @test ibsi_test(radiomic_2D_features["glszm_LargeAreaEmphasis"], 67.0, 0)
            @test ibsi_test(radiomic_2D_features["glszm_LargeAreaHighGrayLevelEmphasis"], 98.66666666666667, 0)
            @test ibsi_test(radiomic_2D_features["glszm_LargeAreaLowGrayLevelEmphasis"], 65.42592592592592, 0)
            @test ibsi_test(radiomic_2D_features["glszm_LowGrayLevelZoneEmphasis"], 0.3634259259259259, 0)
            @test ibsi_test(radiomic_2D_features["glszm_SizeZoneNonUniformity"], 1.0, 0)
            @test ibsi_test(radiomic_2D_features["glszm_SizeZoneNonUniformityNormalized"], 0.3333333333333333, 0)
            @test ibsi_test(radiomic_2D_features["glszm_SmallAreaEmphasis"], 0.4183673469387755, 0)
            @test ibsi_test(radiomic_2D_features["glszm_SmallAreaHighGrayLevelEmphasis"], 13.335034013605442, 0)
            @test ibsi_test(radiomic_2D_features["glszm_SmallAreaLowGrayLevelEmphasis"], 0.016168272864701436, 0)
            @test ibsi_test(radiomic_2D_features["glszm_ZoneEntropy"], 1.5849625007211552, 0)
            @test ibsi_test(radiomic_2D_features["glszm_ZonePercentage"], 0.17647058823529413, 0)
            @test ibsi_test(radiomic_2D_features["glszm_ZoneVariance"], 34.888888888888886, 0)

            # ---- NGTDM ---- (5 features)
            @test ibsi_test(radiomic_2D_features["ngtdm_Busyness"], 3.940625, 0)
            @test ibsi_test(radiomic_2D_features["ngtdm_Coarseness"], 0.13481363996827916, 0)
            @test ibsi_test(radiomic_2D_features["ngtdm_Complexity"], 8.37328431372549, 0)
            @test ibsi_test(radiomic_2D_features["ngtdm_Contrast"], 0.7298857259166641, 0)
            @test ibsi_test(radiomic_2D_features["ngtdm_Strength"], 3.5422771781859765, 0)

            #--- Shape2D --- 
            @test ibsi_test(radiomic_2D_features["shape2d_elongation"], 0.7143974655883708, 0.005)
            @test ibsi_test(radiomic_2D_features["shape2d_major_axis_length"], 11.63955229821705, 0.1)
            @test ibsi_test(radiomic_2D_features["shape2d_maximum_diameter"], 11.661903789690601, 0.1)
            @test ibsi_test(radiomic_2D_features["shape2d_mesh_surface"], 68.0, 4)
            @test ibsi_test(radiomic_2D_features["shape2d_minor_axis_length"], 8.315266662429558, 0.06)
            @test ibsi_test(radiomic_2D_features["shape2d_perimeter"], 38.14213562373095, 0)
            @test ibsi_test(radiomic_2D_features["shape2d_perimeter_surface_ratio"], 0.560913759172514, 0.004)
            @test ibsi_test(radiomic_2D_features["shape2d_pixel_surface"], 68.0, 0)
            @test ibsi_test(radiomic_2D_features["shape2d_sphericity"], 0.7663980345420807, 0.005)
            @test ibsi_test(radiomic_2D_features["shape2d_number_of_islands"], 1, 0)

        end

    end

end