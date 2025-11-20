using Test
using NPZ
using Radiomics

@testset "Radiomics test - GLCM Features" begin

    # Load the prepared data
    data = npzread("sample_data/lungs_data.npz")
    img = data["image"]
    mask = data["mask"]

    # Dummy spacing - replace with actual spacing if needed
    spacing = [1.0, 1.0, 1.0]

    # Extract features
    radiomic_features = Radiomics.extract_radiomic_features(img, mask, spacing; verbose = false)

    # ---- Test delle 24 GLCM features ----
    # Nota: JointAverage ha uno spazio extra nel nome del test originale
    
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


    
end