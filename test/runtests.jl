using Test

using NIfTI
using Radiomics

@testset "Radiomics test" begin

    ct = niread("../sample_data/CTChest.nii.gz")
    mask = niread("../sample_data/Lungs.nii.gz")

    radiomic_features = Radiomics.extract_radiomic_features(ct, mask; binarize_mask = true, verbose = false)

    @test isapprox(radiomic_features["firstorder_entropy"]                        , 2.7950206f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_energy"]                         , 3.835767f11; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_median"]                         , -867.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_total_energy"]                   , 4.3152378f12; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_standard_deviation"]             , 51.536602f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_range"]                          , 366.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_interquartile_range"]            , 54.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_mean"]                           , -854.6469f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_skewness"]                       , 1.4568754f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_mean_absolute_deviation"]        , 37.97294f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_uniformity"]                     , 0.1843532f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile90"]                   , -786.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_minimum"]                        , -1019.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_root_mean_squared"]              , 856.1967f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_variance"]                       , 2654.853f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_percentile10"]                   , -904.0f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_robust_mean_absolute_deviation"] , 23.399605f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_kurtosis"]                       , 5.393815f0; atol=1e-4)
    @test isapprox(radiomic_features["firstorder_maximum"]                        , -653.0f0; atol=1e-4)

end

