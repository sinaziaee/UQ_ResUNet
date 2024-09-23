import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loader import KitsDataset, CombinedDataset
from torch.utils.data import Dataset
from res_unet import ResidualUNet
from tqdm import tqdm
import datetime
import time
from utils import MultiChannelMaskTransform, PolyLRScheduler
from utils import validate_epoch, combined_loss, ema_update, load_model, get_last_checkpoint, create_semi_train_data
from uncert_utils import find_teacher_models_pahts, load_teacher_models, teachers_perdict, teachers_predict2
import configs as configs
import warnings
import numpy as np
import matplotlib.pyplot as plt
import shutil

warnings.filterwarnings('ignore')
print("semi_train3.py")

def train(train_kind, save_path, params, val_base_dir, unlabeled_base_dir):
    os.makedirs(save_path, exist_ok=True)
    # Hyperparameters
    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]
    power = params["power"]
    batch_size = params["batch_size"]
    transform = params["transform"]

    labeled_images_dir, labeled_segmentations_dir = create_semi_train_data(train_kind)
    print("Finished Creating temporary semi_train data")
    val_images_dir, val_segmentations_dir = os.path.join(val_base_dir, 'images'), os.path.join(val_base_dir, 'segmentations')
    unlabeled_images_dir, unlabeled_segmentation_dir = os.path.join(unlabeled_base_dir, 'images'), os.path.join(unlabeled_base_dir, 'segmentations')
    teacher_models_paths = find_teacher_models_pahts()
    teacher_models = load_teacher_models(teacher_models_paths)
    device = torch.device("cuda")

    unlabeled_dataset = KitsDataset(images_dir=unlabeled_images_dir, segmentations_dir=unlabeled_segmentation_dir, transform=transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=4)
    valid_dataset = KitsDataset(images_dir=val_images_dir, segmentations_dir=val_segmentations_dir, transform=transform)
    labeled_dataset = KitsDataset(images_dir=labeled_images_dir, segmentations_dir=labeled_segmentations_dir, transform=transform)
    teacher_model = teacher_models[1]
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    student_model = ResidualUNet(in_channels=1, num_classes=configs.NUM_CLASSES)
    student_model.load_state_dict(teacher_model.state_dict())  # Copy teacher weights to student
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = PolyLRScheduler(optimizer, max_epochs=num_epochs, power=power)
    best_val_loss = float('inf')
    metrics_file_path = os.path.join(save_path, 'training_metrics.txt')
    student_model = student_model.to(device)
    print("device:", device)
    with open(metrics_file_path, 'w') as f:
        f.write(f"{'Epoch':<8} | {'Time (s)':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<10}\n")
        f.write("-" * 120 + "\n")
        f.flush()
        
        for epoch in tqdm(range(num_epochs)):
            start_time = time.time()
            pseudo_labels_dir, ece_scores, entropy_scores = teachers_predict2(teacher_models, unlabeled_loader, device)
            # del unlabeled_loader
            # for teacher_model in teacher_models:
            #     teacher_model = teacher_model.to('cpu')
            
            print(f"Pseudo labels generated for epoch {epoch + 1}!")
            pseudo_labels_dataset = KitsDataset(images_dir=unlabeled_images_dir, segmentations_dir=pseudo_labels_dir, transform=transform)
            combined_dataset = CombinedDataset(labeled_dataset, unlabeled_dataset, pseudo_labels_dataset)
            train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            # Training the student model
            student_model = student_model.to(device)
            student_model.train()
            running_loss = 0.0
            for i, (images, masks) in tqdm(enumerate(train_loader)):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                
                student_outputs = student_model(images)
                supervised_loss = combined_loss(student_outputs, masks)
                
                loss = supervised_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            val_loss, val_acc, val_dice, val_iou = validate_epoch(student_model, val_loader, combined_loss, device)

            epoch_time = time.time() - start_time
            f.write(f"{epoch+1:<8} | {epoch_time:<10.2f} | {running_loss/len(train_loader):<12.4f} | {val_loss:<10.4f} | {val_dice:<10.4f} | {val_iou:<10.4f}\n")
            f.flush()

            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f} sec")
            print(f"Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(student_model.state_dict(), os.path.join(save_path, 'best_student_model.pth'))
                print("Best student model saved!")
            
            # Update teacher models with EMA
            teacher_models = ema_update(teacher_models, student_model.to('cpu'))
            # Step the scheduler
            scheduler.step()
            # del train_loader
            student_model = student_model.to('cpu')
            del combined_dataset
            del pseudo_labels_dataset
    shutil.rmtree(os.path.join(configs.base_processed_path_dir, "semi_train"))
    shutil.rmtree(os.path.join(configs.base_processed_path_dir, "pseudo_labels"))


if __name__ == "__main__":

    save_path = os.path.join(configs.base_analysis_result_dir, 'semi_supervised', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    train_kind = "train"
    unlabeled_kind = "unlabeled"
    # Hyperparameters
    num_epochs = 50
    learning_rate = 1e-4
    power = 0.9
    batch_size = configs.BATCH_SIZE
    transform = MultiChannelMaskTransform(
        A.Compose([
            A.GaussianBlur(p=0.3),    
            A.Normalize(mean=(0.0,), std=(1.0,)),  
            ToTensorV2()
        ])
    )
    params = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "power": power,
        "batch_size": batch_size,
        "transform": transform
    }
    val_base_dir = os.path.join(configs.base_processed_path_dir, 'valid')
    unlabeled_base_dir = os.path.join(configs.base_processed_path_dir, unlabeled_kind)

    train(train_kind, save_path, params, val_base_dir, unlabeled_base_dir)