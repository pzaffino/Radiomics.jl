using Test
using NPZ
using Radiomics

@testset "Radiomics test - NGTDM Features" begin

    # Load the prepared data
    data = npzread("sample_data/lungs_data.npz")
    img = data["image"]
    mask = data["mask"]

    # Dummy spacing - replace with actual spacing if needed
    spacing = [1.0, 1.0, 1.0]

    # Extract features
    radiomic_features = Radiomics.extract_radiomic_features(img, mask, spacing; verbose = false)

    @test isapprox(radiomic_features["ngtdm_Busyness"], 622.092115405341, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Coarseness"], 1.24647901126327e-05, atol=1e-5)
    @test isapprox(radiomic_features["ngtdm_Complexity"], 114.354389990473, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Contrast"], 0.0483421505790213, rtol=1e-4)
    @test isapprox(radiomic_features["ngtdm_Strength"], 0.00114419088561417, rtol=1e-4)

end
