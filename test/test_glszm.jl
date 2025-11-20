using Test
using NPZ
using Radiomics

@testset "Radiomics test - GLSZM Features" begin

    # Load the prepared data
    data = npzread("sample_data/lungs_data.npz")
    img = data["image"]
    mask = data["mask"]

    # Dummy spacing - replace with actual spacing if needed
    spacing = [1.0, 1.0, 1.0]

    # Extract features
    radiomic_features = Radiomics.extract_radiomic_features(img, mask, spacing; verbose = false)

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

end
