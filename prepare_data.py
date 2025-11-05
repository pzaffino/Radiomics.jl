import nibabel as nib
import numpy as np

def prepare_data():
    """
    Loads the CTChest and Lungs NIfTI files, and saves the image and mask data as a .npz file.
    """
    ct_image = nib.load("sample_data/CTChest.nii.gz")
    lungs_mask = nib.load("sample_data/Lungs.nii.gz")

    ct_data = ct_image.get_fdata()
    lungs_data = lungs_mask.get_fdata()

    np.savez("sample_data/lungs_data.npz", image=ct_data, mask=lungs_data)

if __name__ == "__main__":
    prepare_data()
