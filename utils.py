import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn 
from torch.optim.lr_scheduler import _LRScheduler
from monai.losses import DiceLoss
from res_unet import ResidualUNet
import os
import nibabel as nib
import configs
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
import shutil
import json

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice[:, 1:].mean()

def dice_coefficient_monai(pred, target, include_background=False):
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    pred = torch.softmax(pred, dim=1)
    dice = dice_metric(pred, target)
    return dice.mean()

# Define the IoU
def iou_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou[:, 1:].mean()

class MultiChannelMaskTransform:
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, image, mask):
        transformed = self.transform(image=image)
        image = transformed['image']
        return {'image': image, 'mask': torch.tensor(mask, dtype=torch.float32)}
    
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    correct = 0
    total = 0
    
    for inx, (images, masks) in tqdm(enumerate(dataloader)):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dice_score += dice_coefficient(outputs, masks).item()
        # dice_score += dice_coefficient_monai(outputs, masks).item()
        iou_score += iou_coefficient(outputs, masks).item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == masks).sum().item()
        total += np.prod(masks.size())

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = dice_score / len(dataloader)
    epoch_iou = iou_score / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, epoch_dice, epoch_iou

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            try:
                outputs = model(images)
            except Exception as e:
                print("Error ---------------------------------------------")
                print(e)
                print(images.shape)
                print(masks.shape)
                print(images.dtype)
                print(masks.dtype)
                continue
            
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            dice_score += dice_coefficient(outputs, masks).item()
            # dice_score += dice_coefficient_monai(outputs, masks).item()
            iou_score += iou_coefficient(outputs, masks).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == masks).sum().item()
            total += np.prod(masks.size())

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = dice_score / len(dataloader)
    epoch_iou = iou_score / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, epoch_dice, epoch_iou

def dice_loss(pred, target, smooth=1e-6, epsilon=1e-8):
    pred = torch.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth + epsilon)  # Adding epsilon to the denominator
    dice_loss = 1 - dice.mean()
    return dice_loss

def combined_loss(pred, target, ce_weight=0.5, dice_weight=0.5, epsilon=1e-8):
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    d_loss = dice_loss(pred, target, epsilon=epsilon)
    return ce_weight * ce_loss + dice_weight * d_loss

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super(PolyLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power
                for base_lr in self.base_lrs]

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def combined_loss_monai(pred, target, ce_weight=0.5, dice_weight=0.5):
    # Initialize the DiceLoss
    dice_loss = DiceLoss(include_background=True, to_onehot_y=True, sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
    # CrossEntropyLoss for comparison
    ce_loss = torch.nn.CrossEntropyLoss()
    # Calculate Dice Loss using MONAI
    d_loss = dice_loss(pred, target)
    # Calculate CrossEntropyLoss
    ce_loss_value = ce_loss(pred, target)
    # Combine the two losses
    return ce_weight * ce_loss_value + dice_weight * d_loss

def load_model(model_path, device, in_channels=1, num_classes=configs.NUM_CLASSES):
    model = ResidualUNet(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def create_segmentation_folder(segmentation_dir, case_name):
    os.makedirs(os.path.join(segmentation_dir, case_name), exist_ok=True)

def get_image_info(original_segment_dir, result_segment_dir=configs.base_inference_dir):
    depths = []
    case_names = []
    affine_matrix_list = []
    for case_name in os.listdir(original_segment_dir):
        # create_segmentation_folder(result_segment_dir, case_name)
        case_number = int(case_name.split('_')[-1])
        image = nib.load(os.path.join(original_segment_dir, case_name, 'segmentation.nii.gz'))
        affine_matrix = image.affine
        affine_matrix_list.append(affine_matrix)
        depths.append(image.shape[0])
        case_names.append(case_name)
        del image, affine_matrix
    return depths, case_names, affine_matrix_list

def class_specific_dice_and_iou_calculator(pred, target, smooth=1e-6):
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    pred_soft = torch.softmax(pred, dim=1)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    dice_scores = []
    iou_scores = []

    # We skip the background class (index 0) and calculate for kidney, tumor, and cyst
    for class_idx in range(1, pred.shape[1]):
        pred_class = pred_soft[:, class_idx:class_idx + 1, :, :]  # Isolate the class channel
        target_class = target[:, class_idx:class_idx + 1, :, :]  # Isolate the corresponding target class channel
        
        dice_metric.reset()
        dice = dice_metric(pred_class, target_class)
        dice_scores.append(dice.mean().item())

        pred_class = pred[:, class_idx, :, :]
        target_class = target[:, class_idx, :, :]
        # Dice Coefficient
        intersection = (pred_class * target_class).sum(dim=(1, 2))
        # union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
        # dice = (2.0 * intersection + smooth) / (union + smooth)
        # dice_scores.append(dice.mean().item())

        # IoU Coefficient
        iou_union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2)) - intersection
        iou = (intersection + smooth) / (iou_union + smooth)
        iou_scores.append(iou.mean().item())

    return dice_scores, iou_scores


def save_segmentation(mask, save_path, slice_num):
    new_mask = mask.squeeze().cpu().numpy().astype(np.uint8)
    new_mask = 255 - new_mask
    
    kidney_path = os.path.join(save_path, "kidney")
    tumor_path = os.path.join(save_path, "tumor")
    cyst_path = os.path.join(save_path, "cyst")
    combined_path = os.path.join(save_path, "combined")
    
    os.makedirs(kidney_path, exist_ok=True)
    os.makedirs(tumor_path, exist_ok=True)
    os.makedirs(cyst_path, exist_ok=True)
    os.makedirs(combined_path, exist_ok=True)
    
    combined_mask = np.zeros_like(new_mask[0], dtype=np.uint8)
        
    mask_kidney = new_mask[1]
    Image.fromarray(new_mask[1], mode='P').save(os.path.join(save_path, kidney_path, f"{slice_num}.png"))
    # plt.imsave(os.path.join(save_path, kidney_path, f"{slice_num}.png"), mask_kidney, cmap='gray')

    mask_tumor = new_mask[2]
    Image.fromarray(new_mask[2], mode='P').save(os.path.join(save_path, tumor_path, f"{slice_num}.png"))
    # plt.imsave(os.path.join(save_path, tumor_path, f"{slice_num}.png"), mask_tumor, cmap='gray')

    mask_cyst = new_mask[3]
    Image.fromarray(new_mask[3], mode='P').save(os.path.join(save_path, cyst_path, f"{slice_num}.png"))
    # plt.imsave(os.path.join(save_path, cyst_path, f"{slice_num}.png"), mask_cyst, cmap='gray')
    
    cyst_image_map = np.where(mask_cyst > 128, 3, 0)
    tumor_image_map = np.where(mask_tumor > 128, 2, 0)
    kid_image_map = np.where(mask_kidney > 128, 1, 0)
    
    for i in range(mask_kidney.shape[0]):
        for j in range(mask_kidney.shape[1]):
            if mask_kidney[i, j] > mask_tumor[i, j] and mask_kidney[i, j] > mask_cyst[i, j] and kid_image_map[i, j] != 0:
                combined_mask[i, j] = 1
            elif mask_tumor[i, j] > mask_cyst[i, j] and mask_tumor[i, j] > mask_kidney[i, j] and tumor_image_map[i, j] != 0:
                combined_mask[i, j] = 2
            elif mask_cyst[i, j] > mask_tumor[i, j] and mask_cyst[i, j] > mask_kidney[i, j] and cyst_image_map[i, j] != 0:
                combined_mask[i, j] = 3
            else:
                combined_mask[i, j] = 0
    # save the combined mask with numpy save
    np.save(os.path.join(save_path, combined_path, f"{slice_num}.npy"), combined_mask)
    
        
def save_iou_dice_results_per_case(case_names, dice_list, iou_list, class_dice_list, class_iou_list, depth_list, output_dir):
    total_dice = np.array(dice_list)
    total_iou = np.array(iou_list)
    
    class_dice = np.array(class_dice_list)
    class_iou = np.array(class_iou_list)
        
    kidney_dice, tumor_dice, cyst_dice  = class_dice[:, 0], class_dice[:, 1], class_dice[:, 2]
    kidney_iou, tumor_iou, cyst_iou = class_iou[:, 0], class_iou[:, 1], class_iou[:, 2]
    
    t_dice, t_iou, t_kid_dice, t_kid_iou, t_tumor_dice, t_tumor_iou, t_cyst_dice, t_cyst_iou = np.copy(total_dice), np.copy(total_iou), np.copy(kidney_dice), np.copy(kidney_iou), np.copy(tumor_dice), np.copy(tumor_iou), np.copy(cyst_dice), np.copy(cyst_iou)
    
    case_dice_list, case_kidney_dice_list, case_tumor_dice_list, case_cyst_dice_list = [], [], [], []
    case_iou_list, case_kidney_iou_list, case_tumor_iou_list, case_cyst_iou_list = [], [], [], []
    
    for inx, depth in enumerate(depth_list):
        case_name = case_names[inx]
        case_dice_list.append(total_dice[:depth].mean())
        case_iou_list.append(total_iou[:depth].mean())
        case_kidney_dice_list.append(kidney_dice[:depth].mean())
        case_kidney_iou_list.append(kidney_iou[:depth].mean())
        case_tumor_dice_list.append(tumor_dice[:depth].mean())
        case_tumor_iou_list.append(tumor_iou[:depth].mean())
        case_cyst_dice_list.append(cyst_dice[:depth].mean())
        case_cyst_iou_list.append(cyst_iou[:depth].mean())
        
        total_dice, kidney_dice, tumor_dice, cyst_dice = total_dice[depth:], kidney_dice[depth:], tumor_dice[depth:], cyst_dice[depth:]
        total_iou, kidney_iou, tumor_iou, cyst_iou = total_iou[depth:], kidney_iou[depth:], tumor_iou[depth:], cyst_iou[depth:]
        
        print(f"Case: {case_name}, Dice: {case_dice_list[inx]:.4f}, IoU: {case_iou_list[inx]:.4f}\n", 
            f"Kidney Dice: {case_kidney_dice_list[inx]:.4f}, Kidney IoU: {case_kidney_iou_list[inx]:.4f}\n",
            f"Tumor Dice: {case_tumor_dice_list[inx]:.4f}, Tumor IoU: {case_tumor_iou_list[inx]:.4f}\n",
            f"Cyst Dice: {case_cyst_dice_list[inx]:.4f}, Cyst IoU: {case_cyst_iou_list[inx]:.4f}\n")
        
    all_classes_average_dice, all_classes_average_iou = np.mean(t_dice), np.mean(t_iou)
    all_classes_average_kidney_dice, all_classes_average_kidney_iou = np.mean(t_kid_dice), np.mean(t_kid_iou)
    all_classes_average_tumor_dice, all_classes_average_tumor_iou = np.mean(t_tumor_dice), np.mean(t_tumor_iou)
    all_classes_average_cyst_dice, all_classes_average_cyst_iou = np.mean(t_cyst_dice), np.mean(t_cyst_iou)
    
    print("---------------------------------------------------------------")
    print(f"Average Dice Score: {all_classes_average_dice:.4f}, Average IoU: {all_classes_average_iou:.4f}")
    print(f"Average Kidney Dice Score: {all_classes_average_kidney_dice:.4f}, Average Kidney IoU: {all_classes_average_kidney_iou:.4f}")
    print(f"Average Tumor Dice Score: {all_classes_average_tumor_dice:.4f}, Average Tumor IoU: {all_classes_average_tumor_iou:.4f}")
    print(f"Average Cyst Dice Score: {all_classes_average_cyst_dice:.4f}, Average Cyst IoU: {all_classes_average_cyst_iou:.4f}")

    results = pd.DataFrame({
        'Case': case_names,
        'Dice': case_dice_list,
        'IoU': case_iou_list,
        'Kidney Dice': case_kidney_dice_list,
        'Kidney IoU': case_kidney_iou_list,
        'Tumor Dice': case_tumor_dice_list,
        'Tumor IoU': case_tumor_iou_list,
        'Cyst Dice': case_cyst_dice_list,
        'Cyst IoU': case_cyst_iou_list
    })
    
    new_row ={
        'Case': 'Average',
        'Dice': all_classes_average_dice,
        'IoU': all_classes_average_iou,
        'Kidney Dice': all_classes_average_kidney_dice,
        'Kidney IoU': all_classes_average_kidney_iou,
        'Tumor Dice': all_classes_average_tumor_dice,
        'Tumor IoU': all_classes_average_tumor_iou,
        'Cyst Dice': all_classes_average_cyst_dice,
        'Cyst IoU': all_classes_average_cyst_iou
    }
    new_row_df = pd.DataFrame([new_row])
    results = pd.concat([results, new_row_df], ignore_index=True)
    results.to_csv(os.path.join(output_dir, 'inference_results.csv'), index=False)
        
def get_last_checkpoint(model_dir="results"):
    checkpoints = sorted(os.listdir(model_dir))
    checkpoints = [c for c in checkpoints if not c.endswith(".csv") and not c.startswith("semi")]
    checkpoints.sort()
    return os.path.join(model_dir, checkpoints[-1]) 

def get_list_of_checkpionts(model_dir="results"):
    checkpoints = sorted(os.listdir(model_dir))
    checkpoints = [c for c in checkpoints if not c.endswith(".csv")]
    checkpoints.sort()
    return [os.path.join(model_dir, c) for c in checkpoints]

def map_segmentation_to_segmentation_folder(depth_list, case_names, segment_slice_number):
    counter = 0
    for i, d in enumerate(depth_list):
        if counter + d > segment_slice_number:
            return case_names[i] , segment_slice_number - counter
        counter += d
    return None

def ema_update(teacher_model, student_model, alpha=0.99):
    for teacher_params, student_params in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_params.data = alpha * teacher_params.data + (1 - alpha) * student_params.data

def create_fold_data(fold, base_processed_dir, kind):
    processed_dir = os.path.join(base_processed_dir, kind)
    with open(os.path.join(processed_dir, 'folds_map.json'), 'r') as f:
        folds_map = json.load(f)
    images_dir = os.path.join(processed_dir)
    segmentations_dir = os.path.join(processed_dir)
    train_cases = folds_map[fold]['train']
    val_cases = folds_map[fold]['val']
    train_images_dir = os.path.join(base_processed_dir, fold, 'train', 'images')
    train_segmentations_dir = os.path.join(base_processed_dir, fold, 'train', 'segmentations')
    val_images_dir = os.path.join(base_processed_dir, fold, 'val', 'images')
    val_segmentations_dir = os.path.join(base_processed_dir, fold, 'val', 'segmentations')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_segmentations_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_segmentations_dir, exist_ok=True)
    for case in tqdm(train_cases):
        image_path_dir = os.path.join(images_dir, case, "images")
        segmentatation_path_dir = os.path.join(segmentations_dir, case, "segmentations")
        for image in os.listdir(image_path_dir):
            shutil.copy(os.path.join(image_path_dir, image), os.path.join(train_images_dir, image))
        for segmentation in os.listdir(segmentatation_path_dir):
            shutil.copy(os.path.join(segmentatation_path_dir, segmentation), os.path.join(train_segmentations_dir, segmentation))
    for case in tqdm(val_cases):
        image_path_dir = os.path.join(images_dir, case, "images")
        segmentatation_path_dir = os.path.join(segmentations_dir, case, "segmentations")
        for image in os.listdir(os.path.join(images_dir, case, "images")):
            shutil.copy(os.path.join(image_path_dir, image), os.path.join(val_images_dir, image))
        for segmentation in os.listdir(os.path.join(segmentations_dir, case, "segmentations")):
            shutil.copy(os.path.join(segmentatation_path_dir, segmentation), os.path.join(val_segmentations_dir, segmentation))
    print(f"Data for {fold} copied successfully!")
    return train_images_dir, train_segmentations_dir, val_images_dir, val_segmentations_dir
