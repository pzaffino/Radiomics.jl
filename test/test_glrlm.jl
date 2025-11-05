using Test
using NPZ
using Radiomics

@testset "Radiomics test - GLRLM Features" begin

    # Load the prepared data
    data = npzread("sample_data/lungs_data.npz")
    img = data["image"]
    mask = data["mask"]

    # Dummy spacing - replace with actual spacing if needed
    spacing = [1.0, 1.0, 1.0]

    # Extract features
    radiomic_features = Radiomics.extract_radiomic_features(img, mask, spacing; verbose = false)

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

end
