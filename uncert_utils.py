import numpy as np
import os
from scipy.stats import entropy
from res_unet import ResidualUNet
import configs as configs
from utils import get_last_checkpoint
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

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
    # predictions shape: (num_models, num_classes, height, width)
    avg_predictions = np.mean(predictions, axis=0)  # Averaging over models
    # Entropy across classes, for each pixel
    pixel_entropy = entropy(avg_predictions, axis=0)  # shape (height, width)
    return pixel_entropy

def generate_pseudo_labels(predictions, get_average=True):
    if np.min(predictions[:, 0] == np.max(predictions[:, 0])):
        predictions = np.zeros_like(predictions)
        predictions = np.mean(predictions, axis=0)
        return predictions
    else:
        if get_average:
            predictions = np.mean(predictions, axis=0)
        # softmax over classes
        probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        # make them between 0 and 1
        probs[0] = (probs[0] - np.min(probs[0])) / (np.max(probs[0]) - np.min(probs[0]))
        probs[1] = (probs[1] - np.min(probs[1])) / (np.max(probs[1]) - np.min(probs[1]))
        probs[2] = (probs[2] - np.min(probs[2])) / (np.max(probs[2]) - np.min(probs[2]))
        probs[3] = (probs[3] - np.min(probs[3])) / (np.max(probs[3]) - np.min(probs[3]))
        # make them binary
        probs[1] = (probs[1] > 0.5).astype(int)
        probs[2] = (probs[2] > 0.5).astype(int)
        probs[3] = (probs[3] > 0.5).astype(int)
        return probs

# Function to calculate ECE
def calculate_ece(predictions, labels, num_bins=15):
    num_classes = predictions.shape[0]  # Number of channels (classes)
    height, width = predictions.shape[1], predictions.shape[2]
    
    ece_per_channel = np.zeros(num_classes)  # Store ECE for each channel

    # Iterate over each channel (class)
    for c in range(num_classes):
        pred_channel = predictions[c]  # Predictions for class c
        label_channel = labels[c]  # Labels for class c
        
        confidences = pred_channel  # In this case, confidences are binary 0 or 1
        accuracies = (pred_channel == label_channel).astype(float)  # 1 if correct, 0 otherwise
        
        ece = 0.0
        total_pixels = height * width  # Total number of pixels in this channel
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)

        # Iterate over bins
        for i in range(num_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.any():  # If there are any pixels in this bin
                bin_confidence = confidences[bin_mask].mean()  # Average confidence in the bin
                bin_accuracy = accuracies[bin_mask].mean()  # Average accuracy in the bin
                ece += (bin_mask.sum() / total_pixels) * np.abs(bin_confidence - bin_accuracy)

        ece_per_channel[c] = ece

    # Return the ECE for each channel (class) and the average ECE across channels
    avg_ece = np.mean(ece_per_channel)
    return ece_per_channel, avg_ece

def teachers_predict2(teacher_models, unlabeled_loader, device):
    pseudo_labels_dir = os.path.join(configs.base_processed_path_dir, 'pseudo_labels')
    os.makedirs(pseudo_labels_dir, exist_ok=True)

    ece_scores = []
    entropy_scores = []
    uncertainty_maps = []

    cpu_device = torch.device("cpu")

    for i, teacher_model in enumerate(teacher_models):      
        teacher_models[i] = teacher_model.to(device)
        teacher_models[i].eval()

    with torch.no_grad():
        for i, (images, segmentations) in tqdm(enumerate(tqdm(unlabeled_loader))):
            images = images.to(device)
            segmentations = segmentations.numpy()
            batch_size = images.size(0)
            batch_predictions = []

            for inx, teacher_model in enumerate(teacher_models):
                # Move the model to GPU for processing
                # teacher_model.to(device)
                outputs = teacher_model(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy()  # Shape: (batch_size, num_classes, H, W)
                batch_predictions.append(outputs)
                # Move the model back to CPU to free up GPU memory
                # teacher_model.to(cpu_device)

            # Convert predictions to the appropriate shape
            batch_predictions = np.stack(batch_predictions, axis=0)  # Shape: (num_models, batch_size, num_classes, H, W)
            batch_predictions = np.transpose(batch_predictions, (1, 0, 2, 3, 4))  # Shape: (batch_size, num_models, num_classes, H, W)

            # Process each image in the batch
            for j in range(batch_size):
                predictions = batch_predictions[j]  # Shape: (num_models, num_classes, H, W)
                segmentation = segmentations[j]  # Shape: (num_classes, H, W)

                uncertainty_map = calculate_entropy(predictions)
                uncertainty_maps.append(uncertainty_map)
                entropy_scores.append(uncertainty_map.mean())
                
                predictions = np.mean(predictions, axis=0)  # Average over models, shape: (num_classes, H, W)
                ece = calculate_ece(predictions, segmentation)
                ece_scores.append(ece)
                pseudo_label = generate_pseudo_labels(predictions, get_average=False)
                # Save the pseudo label
                pseudo_label_path = os.path.join(pseudo_labels_dir, f"pseudo_label_{i * batch_size + j}.npy")
                np.save(pseudo_label_path, pseudo_label)

            # Clean up to free memory
            del batch_predictions
            torch.cuda.empty_cache()

    # After processing, ensure all models are moved to CPU and deleted
    for model in teacher_models:
        model.to(cpu_device)
        del model

    torch.cuda.empty_cache()

    return pseudo_labels_dir, ece_scores, entropy_scores


def teachers_perdict(teacher_models, unlabeled_loader, device):
    pseudo_labels = []
    uncertainty_maps = []
    ece_scores = []
    entropy_scores = []

    with torch.no_grad():
        cpu_device = torch.device("cpu")
        all_models_outputs = []
        for inx, teacher_model in enumerate(teacher_models):
            teacher_model.eval()
            teacher_model.to(device)
            one_model_outputs = []
            for images, segmentations in tqdm(unlabeled_loader):
                images = images.to(device)
                outputs = teacher_model(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get class probabilities
                one_model_outputs.append(outputs.cpu().numpy().squeeze(axis=0))
            # all_models_image_outputs.append(np.stack(model_image_outputs))  # Stack outputs for each model
            one_model_outputs = np.stack(one_model_outputs)
            all_models_outputs.append(one_model_outputs)
            del one_model_outputs
            # unloading the model from the GPU
            teacher_model.to(cpu_device)
            del teacher_model
            print(f"finished model {inx} prediction")
        all_models_outputs = np.stack(all_models_outputs)
        all_models_outputs = np.transpose(all_models_outputs, (1, 0, 2, 3, 4))  # shape (num_images, num_models, num_classes, height, width)
        print("started generating pseudo labels")
    for i, (predictions, (_, segmentations)) in tqdm(enumerate(zip(all_models_outputs, unlabeled_loader))):
        segmentations = segmentations.numpy().squeeze(axis=0)
        
        uncertainty_map = calculate_entropy(predictions) 
        uncertainty_maps.append(uncertainty_map)
        entropy_scores.append(uncertainty_map.mean())

        ece = calculate_ece(predictions, segmentations)
        ece_scores.append(ece)

        pseudo_label = generate_pseudo_labels(predictions)
        pseudo_labels.append(pseudo_label)



    pseudo_labels_dir = os.path.join(configs.base_processed_path_dir, 'pseudo_labels')
    os.makedirs(pseudo_labels_dir, exist_ok=True)
    for i, pseudo_label in enumerate(pseudo_labels):
        pseudo_label_path = os.path.join(pseudo_labels_dir, f"pseudo_label_{i}.npy")
        np.save(pseudo_label_path, pseudo_label)

    # unload and delete everything in the GPU
    for model in teacher_models:
        model.to('cpu')
        del model
    del pseudo_labels
    torch.cuda.empty_cache()

    return pseudo_labels_dir, ece_scores, entropy_scores

def save_uncertainty_maps(mask, save_path, slice_num):
    uncert_path = os.path.join(save_path, "uncertainty")
    os.makedirs(uncert_path, exist_ok=True)
    np.save(os.path.join(save_path, uncert_path, f"{slice_num}.npy"), mask)