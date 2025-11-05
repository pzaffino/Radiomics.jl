import pandas as pd

def extract_features(feature_class):
    """
    Reads the pyradiomics_features.ods file and prints the feature values for the specified class.
    """
    df = pd.read_excel("sample_data/pyradiomics_features.ods", engine="odf")

    features = df[df["Feature Class"] == feature_class]

    for _, row in features.iterrows():
        feature_name = row["Feature Name"]
        value = row["Lungs_segment_Segment_1"]
        print(f'@test isapprox(radiomic_features["{feature_class}_{feature_name}"], {value}, atol=1e-4)')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extract_features(sys.argv[1])
    else:
        print("Please provide a feature class (e.g., glszm, ngtdm)")
