import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import configs
from tqdm import tqdm

base_combined_path = os.path.join(configs.base_inference_dir, configs.test_folder)
original_image_path = os.path.join(configs.base_case_dir, configs.test_folder)

for case_name in tqdm(os.listdir(base_combined_path)):
    if case_name.endswith(".csv"):
        continue
    stack = []
    affine = nib.load(os.path.join(original_image_path, case_name, "segmentation.nii.gz")).affine
    case_path = os.path.join(base_combined_path, case_name, "combined")
    for seg_name in os.listdir(case_path):
        seg_path = os.path.join(case_path, seg_name)
        seg = np.load(seg_path)
        stack.append(seg)
    stack = np.array(stack)
    nifti_path = os.path.join(base_combined_path, case_name, f"{case_name}_prediction.nii.gz")
    nib.save(nib.Nifti1Image(stack, affine), nifti_path)
    print(f"Saved {case_name}_prediction.nii.gz")

    