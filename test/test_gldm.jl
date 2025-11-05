using Test
using NPZ
using Radiomics

@testset "Radiomics test - GLDM Features" begin

    # Load the prepared data
    data = npzread("sample_data/lungs_data.npz")
    img = data["image"]
    mask = data["mask"]

    # Dummy spacing - replace with actual spacing if needed
    spacing = [1.0, 1.0, 1.0]

    # Extract features
    radiomic_features = Radiomics.extract_radiomic_features(img, mask, spacing; verbose = false)

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

end
