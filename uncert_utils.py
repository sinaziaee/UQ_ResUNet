import numpy as np
import os


def calculate_entropy(prob_maps):
    entropies = -np.sum(prob_maps * np.log(prob_maps + 1e-8), axis=0)
    return entropies

def save_uncertainty_maps(mask, save_path, slice_num):
    uncert_path = os.path.join(save_path, "uncertainty")
    os.makedirs(uncert_path, exist_ok=True)
    np.save(os.path.join(save_path, uncert_path, f"{slice_num}.npy"), mask)