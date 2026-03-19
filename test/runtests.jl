using Test
using NIfTI
using Radiomics
using Downloads

"""
IBSI Validation Test Suite

This module contains automated tests to validate the `extract_radiomic_features` 
function against the Image Biomarker Standardisation Initiative (IBSI) guidelines. 
It extracts features from a standardized digital phantom and asserts that the 
computed values strictly match the official IBSI reference values, ensuring the 
mathematical correctness and compliance of the extraction pipeline.

Reference Data:
The benchmark values used for validation in this suite are derived from the 
official IBSI Phase 1 standard. 
The reference manuals and data sets can be accessed at:
- Reference Data Sets Guide: https://ibsi.readthedocs.io/en/latest/05_Reference_data_sets.html
- Official Website: https://theibsi.github.io/
- Data Repository: https://github.com/theibsi/data_sets
"""

const PHANTOM_DIR = "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_1_digital_phantom/nifti/image/phantom.nii.gz"
const MASK_DIR    = "https://raw.githubusercontent.com/theibsi/data_sets/master/ibsi_1_digital_phantom/nifti/mask/mask.nii.gz"

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

mktempdir() do tmpdir
    phantom_path = joinpath(tmpdir, "phantom.nii.gz")
    mask_path = joinpath(tmpdir, "mask.nii.gz")
    @info "Downloading IBSI dataset..."
    
    @info "Downloading phantom..."
    Downloads.download(PHANTOM_DIR, phantom_path)
    @info "Downloading mask..."
    Downloads.download(MASK_DIR, mask_path)

    @info "Download complete."

    @testset "Radiomics test" begin

        ct = niread(phantom_path)
        mask = niread(mask_path)
        spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]

        radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=false, sample_rate = 1.0, n_bins = 6, keep_largest_only=false)

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
        @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformity"], 26.9 , 0)
        @test ibsi_test(radiomic_features["glrlm_RunLengthNonUniformityNormalized"], 0.513, 0)
        @test ibsi_test(radiomic_features["glrlm_RunPercentage"], 0.68, 0)
        @test ibsi_test(radiomic_features["glrlm_RunVariance"], 0.574 , 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunEmphasis"], 0.705, 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunHighGrayLevelEmphasis"], 8.540, 0)
        @test ibsi_test(radiomic_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.352, 0)
        
        # ---- GLSZM ---- (16 features)
        @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformity"], 1.4, 0)
        @test ibsi_test(radiomic_features["glszm_GrayLevelNonUniformityNormalized"], 0.28, 0)
        @test ibsi_test(radiomic_features["glszm_GrayLevelVariance"], 2.64, 0)
        @test ibsi_test(radiomic_features["glszm_HighGrayLevelZoneEmphasis"], 15.6, 0)
        @test ibsi_test(radiomic_features["glszm_LargeAreaEmphasis"], 550, 0)
        @test ibsi_test(radiomic_features["glszm_LargeAreaHighGrayLevelEmphasis"], 1490 , 0)
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

end