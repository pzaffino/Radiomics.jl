include("test_glszm.jl")
include("test_ngtdm.jl")
using Test

using NIfTI
using Radiomics

@testset "Radiomics test - First Order Features" begin

    ct = niread("sample_data/CTChest.nii.gz")
    mask = niread("sample_data/Lungs.nii.gz")
    spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]

    radiomic_features = Radiomics.extract_radiomic_features(ct.raw, mask.raw, spacing; verbose = false)

    @test isapprox(radiomic_features["firstorder_entropy"]                        , 2.7950206f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_energy"]                         , 3.835767f11; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_median"]                         , -867.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_total_energy"]                   , 4.3152378f12; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_standard_deviation"]             , 51.536602f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_range"]                          , 366.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_interquartile_range"]            , 54.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_mean"]                           , -854.6469f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_skewness"]                       , 1.4568754f0; rtol=3e-4)
    @test isapprox(radiomic_features["firstorder_mean_absolute_deviation"]        , 37.97294f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_uniformity"]                     , 0.1843532f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile90"]                   , -786.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_minimum"]                        , -1019.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_root_mean_squared"]              , 856.1967f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_variance"]                       , 2654.853f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile10"]                   , -904.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_robust_mean_absolute_deviation"] , 23.399605f0; rtol=3e-4)
    @test isapprox(radiomic_features["firstorder_kurtosis"]                       , 5.393815f0; rtol=1e-4)
    @test isapprox(radiomic_features["firstorder_maximum"]                        , -653.0f0; atol=1e-4)
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

    radiomic_features = Radiomics.extract_radiomic_features(ct_slice, mask_slice, spacing; force_2d=true, force_2d_dimension=2, verbose = false)

    @test isapprox(radiomic_features["shape2d_elongation"]                        , 0.6524707f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_major_axis_length"]                 , 271.63801f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_maximum_diameter"]                  , 242.68498f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_mesh_surface"]                      , 26619.75f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_minor_axis_length"]                 , 177.23585f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_perimeter"]                         , 2146.99826f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_perimeter_surface_ratio"]           , 0.08065f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_pixel_surface"]                     , 26556.75f0; atol=1e-4)
    @test isapprox(radiomic_features["shape2d_sphericity"]                        , 0.269386f0; atol=1e-4)
end

@testset "Radiomics test - Shape3D Features" begin
    # Read NIfTI files

    ct = niread("sample_data/CTChest.nii.gz")
    mask = niread("sample_data/Lungs.nii.gz")
    spacing = [ct.header.pixdim[2], ct.header.pixdim[3], ct.header.pixdim[4]]
    spacing_perm = reverse(spacing)

    mask_bin = mask.raw .> 0
    mask_cc = Radiomics.keep_largest_component(mask_bin)

    mask_pad = Radiomics.pad_mask(mask_cc, 1)
    features = Radiomics.get_shape3d_features(mask_pad, spacing; verbose=true)

    @test isapprox(features["shape3d_surface_area"], 590308.1f0; rtol=0.02)
    @test isapprox(features["shape3d_mesh_volume"], 5.9246475e6; rtol=0.015)
    @test isapprox(features["shape3d_surface_volume_ratio"], 0.09963599f0; rtol=0.02)
    @test isapprox(features["shape3d_sphericity"], 0.26823393f0; rtol=0.02)
    @test isapprox(features["shape3d_major_axis_length"], 314.70545f0; atol=0.01)
    @test isapprox(features["shape3d_minor_axis_length"], 240.64522f0; atol=0.01)
    @test isapprox(features["shape3d_least_axis_length"], 180.21617f0; atol=0.01)
    @test isapprox(features["shape3d_elongation"], 0.7646681f0; atol=0.01)
    @test isapprox(features["shape3d_flatness"], 0.5726503f0; atol=0.01)
    @test isapprox(features["shape3d_voxel_volume"], 5.886506e6; atol=0.01)
end
