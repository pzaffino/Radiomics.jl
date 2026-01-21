using Test

using NIfTI
using Radiomics

@testset "Radiomics test" begin

    ct = niread("sample_data/CTChest.nii.gz")
    mask = niread("sample_data/Lungs.nii.gz")
    spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]

    radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose=false)

    @test isapprox(radiomic_features["firstorder_entropy"], 2.833073; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_energy"], 3.835767f11; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_median"], -867.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_total_energy"], 4.3152378f12; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_standard_deviation"], 51.536602f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_range"], 366.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_interquartile_range"], 54.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_mean"], -854.6469f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_skewness"], 1.4568754f0; rtol=3e-4)
    @test isapprox(radiomic_features["firstorder_mean_absolute_deviation"], 37.97294f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_uniformity"], 0.18202744; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile90"], -786.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_minimum"], -1019.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_root_mean_squared"], 856.1967f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_variance"], 2654.853f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile10"], -904.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_robust_mean_absolute_deviation"], 23.399605f0; rtol=3e-4)
    @test isapprox(radiomic_features["firstorder_kurtosis"], 5.393815f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_maximum"], -653.0f0; atol=1e-4)

    # ---- GLCM ----
    @test isapprox(radiomic_features["glcm_Autocorrelation"], 53.23984, atol=1e-4)
    @test isapprox(radiomic_features["glcm_ClusterProminence"], 469.07855, atol=1e-4)
    @test isapprox(radiomic_features["glcm_ClusterShade"], 33.83824, atol=1e-4)
    @test isapprox(radiomic_features["glcm_ClusterTendency"], 10.075768, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Contrast"], 5.269082, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Correlation"], 0.3155505, atol=1e-4)
    @test isapprox(radiomic_features["glcm_DifferenceAverage"], 1.5373107, atol=1e-4)
    @test isapprox(radiomic_features["glcm_DifferenceEntropy"], 2.3570516, atol=1e-4)
    @test isapprox(radiomic_features["glcm_DifferenceVariance"], 2.8628466, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Id"], 0.5538825, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Idm"], 0.5066675, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Idmn"], 0.9791813, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Idn"], 0.91537666, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Imc1"], -0.05739861, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Imc2"], 0.48559535, atol=1e-4)
    @test isapprox(radiomic_features["glcm_InverseVariance"], 0.42546532, atol=1e-4)
    @test isapprox(radiomic_features["glcm_JointAverage"], 7.213738, atol=1e-4)
    @test isapprox(radiomic_features["glcm_JointEnergy"], 0.04378508, atol=1e-4)
    @test isapprox(radiomic_features["glcm_JointEntropy"], 5.3672366, atol=1e-4)
    @test isapprox(radiomic_features["glcm_MaximumProbability"], 0.11300446, atol=1e-4)
    @test isapprox(radiomic_features["glcm_Mcc"], 0.64229226, atol=1e-4)
    @test isapprox(radiomic_features["glcm_SumAverage"], 14.427477, atol=1e-4)
    @test isapprox(radiomic_features["glcm_SumEntropy"], 3.5658517, atol=1e-4)
    @test isapprox(radiomic_features["glcm_SumSquares"], 3.8362124, atol=1e-4)

    # ---- GLDM ----
    @test isapprox(radiomic_features["gldm_DependenceEntropy"], 6.44838151997623, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_DependenceNonUniformity"], 33772.0249997611, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_DependenceNonUniformityNormalized"], 0.0645434261192388, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_DependenceVariance"], 19.8236046617049, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_GrayLevelNonUniformity"], 95244.9530583188, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_GrayLevelVariance"], 4.32280580050045, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_HighGrayLevelEmphasis"], 58.1083201941729, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_LargeDependenceEmphasis"], 78.3759558141979, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_LargeDependenceHighGrayLevelEmphasis"], 3328.04693785894, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_LargeDependenceLowGrayLevelEmphasis"], 2.07139964721161, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_LowGrayLevelEmphasis"], 0.0226096714653867, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_SmallDependenceEmphasis"], 0.11504235840153, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_SmallDependenceHighGrayLevelEmphasis"], 12.1360197444495, rtol=1e-1)
    @test isapprox(radiomic_features["gldm_SmallDependenceLowGrayLevelEmphasis"], 0.00166511826653721, rtol=1e-1)

    # ---- GLRLM ----
    @test isapprox(radiomic_features["glrlm_GrayLevelNonUniformity"], 62334.0772321369, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_GrayLevelNonUniformityNormalized"], 0.159932805139573, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_GrayLevelVariance"], 5.00344601884642, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_HighGrayLevelRunEmphasis"], 62.7111202810391, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_LongRunEmphasis"], 2.42129490956702, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_LongRunHighGrayLevelEmphasis"], 124.927120209026, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_LongRunLowGrayLevelEmphasis"], 0.058524288898688, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_LowGrayLevelRunEmphasis"], 0.0215852447688952, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_RunEntropy"], 4.00702860298145, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_RunLengthNonUniformity"], 241760.714608787, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_RunLengthNonUniformityNormalized"], 0.61795300183296, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_RunPercentage"], 0.744155738192948, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_RunVariance"], 0.594074516046419, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_ShortRunEmphasis"], 0.813463940507583, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_ShortRunHighGrayLevelEmphasis"], 54.0673914593487, rtol=1e-4)
    @test isapprox(radiomic_features["glrlm_ShortRunLowGrayLevelEmphasis"], 0.0169147308013419, rtol=1e-4)

    # ---- GLSZM ----
    @test isapprox(radiomic_features["glszm_GrayLevelNonUniformity"], 6302.47206834876, atol=1e-2)
    @test isapprox(radiomic_features["glszm_GrayLevelNonUniformityNormalized"], 0.109442618444245, atol=1e-2)
    @test isapprox(radiomic_features["glszm_GrayLevelVariance"], 7.47128144303462, atol=1e-2)
    @test isapprox(radiomic_features["glszm_HighGrayLevelZoneEmphasis"], 115.719502665532, atol=1e-2)
    @test isapprox(radiomic_features["glszm_LargeAreaEmphasis"], 500306.527323875, atol=1e-2)
    @test isapprox(radiomic_features["glszm_LargeAreaHighGrayLevelEmphasis"], 22272682.0047928, atol=1e-2)
    @test isapprox(radiomic_features["glszm_LargeAreaLowGrayLevelEmphasis"], 11699.2163584262, atol=1e-2)
    @test isapprox(radiomic_features["glszm_LowGrayLevelZoneEmphasis"], 0.012881060611444, atol=1e-2)
    @test isapprox(radiomic_features["glszm_SizeZoneNonUniformity"], 27063.7007831629, atol=1e-2)
    @test isapprox(radiomic_features["glszm_SizeZoneNonUniformityNormalized"], 0.469961984183286, atol=1e-2)
    @test isapprox(radiomic_features["glszm_SmallAreaEmphasis"], 0.712902358873614, atol=1e-2)
    @test isapprox(radiomic_features["glszm_SmallAreaHighGrayLevelEmphasis"], 86.1041377637599, atol=1e-2)
    @test isapprox(radiomic_features["glszm_SmallAreaLowGrayLevelEmphasis"], 0.00879998080156031, atol=1e-2)
    @test isapprox(radiomic_features["glszm_ZoneEntropy"], 5.10082803485719, atol=1e-2)
    @test isapprox(radiomic_features["glszm_ZonePercentage"], 0.110057430075777, atol=1e-2)
    @test isapprox(radiomic_features["glszm_ZoneVariance"], 500223.968924373, atol=1e-2)

    # ---- NGTDM ----
    @test isapprox(radiomic_features["ngtdm_Busyness"], 622.092115405341, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Coarseness"], 1.24647901126327e-05, atol=1e-5)
    @test isapprox(radiomic_features["ngtdm_Complexity"], 114.354389990473, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Contrast"], 0.0483421505790213, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Strength"], 0.00114419088561417, rtol=1e-4)

    #--- Shape3D ---
    # With sample_rate = 0.03, default settings
    @test isapprox(radiomic_features["shape3d_surface_area"], 590308.1f0; rtol=0.02)
    @test isapprox(radiomic_features["shape3d_mesh_volume"], 5.9246475e6; rtol=0.015)
    @test isapprox(radiomic_features["shape3d_surface_volume_ratio"], 0.09963599f0; rtol=0.02)
    @test isapprox(radiomic_features["shape3d_sphericity"], 0.26823393f0; rtol=0.02)
    @test isapprox(radiomic_features["shape3d_major_axis_length"], 314.70545f0; atol=0.01)
    @test isapprox(radiomic_features["shape3d_maximum_3d_diameter"], 345.69965; atol=0.04)
    @test isapprox(radiomic_features["shape3d_minor_axis_length"], 240.64522f0; atol=0.01)
    @test isapprox(radiomic_features["shape3d_least_axis_length"], 180.21617f0; atol=0.01)
    @test isapprox(radiomic_features["shape3d_elongation"], 0.7646681f0; atol=0.01)
    @test isapprox(radiomic_features["shape3d_flatness"], 0.5726503f0; atol=0.01)
    @test isapprox(radiomic_features["shape3d_voxel_volume"], 5.886506e6; atol=0.01)

    # Test explicit sample_rate (High Precision), 349.3029 was observed in manual testing for sample_rate=1.0
    features_precise = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; features=[:shape3d], sample_rate=1.0, verbose=false)
    @test isapprox(features_precise["shape3d_maximum_3d_diameter"], 349.3029f0; atol=0.1)

end

@testset "Radiomics test - Shape2D Features" begin

    ct = niread("sample_data/CTChest.nii.gz")
    mask = niread("sample_data/Lungs.nii.gz")
    spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]

    # Extract a single slice for Shape2D features
    slice_idx = 51
    """
    Python equivalent using SimpleITK to reproduce the results:
    ct_slice = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct)[slice_idx-1,:,:])
    mask_slice = sitk.GetImageFromArray(sitk.GetArrayFromImage(mask)[slice_idx-1,:,:])
    """
    ct_slice = ct.raw[:, :, slice_idx]
    mask_slice = mask.raw[:, :, slice_idx]

    radiomic_features = Radiomics.extract_radiomic_features(ct_slice, mask_slice, spacing[1:2]; verbose=false)

    @test isapprox(radiomic_features["shape2d_elongation"], 0.6524707f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_major_axis_length"], 271.63801f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_maximum_diameter"], 242.68498f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_mesh_surface"], 26619.75f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_minor_axis_length"], 177.23585f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_perimeter"], 2146.99826f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_perimeter_surface_ratio"], 0.08065f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_pixel_surface"], 26556.75f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_sphericity"], 0.269386f0; atol=1e-4)
end