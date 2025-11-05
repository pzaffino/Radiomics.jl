import pandas as pd

def extract_glszm_features():
    """
    Reads the pyradiomics_features.ods file and prints the GLSZM feature values.
    """
    df = pd.read_excel("sample_data/pyradiomics_features.ods", engine="odf")

    glszm_features = df[df["Feature Class"] == "glszm"]

    for _, row in glszm_features.iterrows():
        feature_name = row["Feature Name"]
        value = row["Lungs_segment_Segment_1"]
        print(f"@test isapprox(radiomic_features[\"glszm_{feature_name}\"], {value}, atol=1e-4)")

if __name__ == "__main__":
    extract_glszm_features()
