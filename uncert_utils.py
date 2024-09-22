import numpy as np
import os
from scipy.stats import entropy
from res_unet import ResidualUNet
import configs as configs
from utils import get_last_checkpoint
import torch

def find_teacher_models_pahts():
    teacher_models_paths = []
    for fold_path in os.listdir(configs.base_analysis_result_dir):
        if fold_path.startswith("fold"):
            fold_path = os.path.join(configs.base_analysis_result_dir, fold_path)
            last_checkpoint_dir = get_last_checkpoint(fold_path)
            fold_path = os.path.join(fold_path, last_checkpoint_dir)
            last_checkpoint_model_path = os.path.join(fold_path, "best_model.pth")
            teacher_models_paths.append(last_checkpoint_model_path)
    return teacher_models_paths

def load_teacher_models(teacher_models_paths):
    teacher_models = []
    for teacher_model_path in teacher_models_paths:
        teacher_model = ResidualUNet(in_channels=1, num_classes=configs.NUM_CLASSES)
        teacher_model.load_state_dict(torch.load(teacher_model_path))
        teacher_models.append(teacher_model)
    return teacher_models


def calculate_entropy(predictions):
    # predictions shape: (num_models, num_classes, channels, height, width)
    avg_predictions = np.mean(predictions, axis=0)  # Averaging over models
    channel_entropy = np.zeros(avg_predictions.shape[1:])  # shape (channels, height, width)
    
    for ch in range(avg_predictions.shape[0]):  # Iterate over channels
        # Calculate entropy per channel
        channel_entropy[ch] = entropy(avg_predictions[:, ch, :, :], base=2, axis=0)  # Entropy for each pixel in channel
    
    return np.mean(channel_entropy, axis=0) 


def calculate_ece(predictions, labels, num_bins=15):
    avg_predictions = np.mean(predictions, axis=0)  # Average predictions over models
    ece_total = 0.0
    total_pixels = 0

    # Iterate over channels
    for ch in range(avg_predictions.shape[0]):  # shape (channels, height, width)
        confidences = np.max(avg_predictions[ch], axis=0)  # Take max confidence per pixel in channel
        pred_classes = np.argmax(avg_predictions[ch], axis=0)  # Predicted classes per pixel in channel

        # Ensure the labels match the shape of the predictions for this channel
        channel_labels = labels[ch]  # shape (height, width)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        pixels_in_channel = confidences.size
        total_pixels += pixels_in_channel

        # Iterate over confidence bins
        for i in range(num_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.any():  # Check if the bin has any pixels
                bin_confidence = confidences[bin_mask]
                bin_pred_classes = pred_classes[bin_mask]
                bin_labels = channel_labels[bin_mask]

                # Calculate accuracy for this bin
                bin_accuracy = (bin_pred_classes == bin_labels).mean()  # Mean of correct predictions
                bin_conf = bin_confidence.mean()  # Average confidence in this bin
                ece += (bin_mask.sum() / pixels_in_channel) * np.abs(bin_conf - bin_accuracy)
        
        ece_total += ece * pixels_in_channel  # Weighted ECE per channel

    return ece_total / total_pixels  # Normalize by total pixels across all channels




def save_uncertainty_maps(mask, save_path, slice_num):
    uncert_path = os.path.join(save_path, "uncertainty")
    os.makedirs(uncert_path, exist_ok=True)
    np.save(os.path.join(save_path, uncert_path, f"{slice_num}.npy"), mask)