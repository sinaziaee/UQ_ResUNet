import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loader import KitsDataset
from sklearn.metrics import jaccard_score
from res_unet import ResidualUNet
from unet import UNet
from tqdm import tqdm 
import datetime
import time
from utils import MultiChannelMaskTransform, PolyLRScheduler
from utils import train_epoch, validate_epoch, combined_loss, combined_loss_monai, initialize_weights, create_fold_data
import configs as configs
import warnings
import shutil
warnings.filterwarnings('ignore')

def train(fold, train_images_dir, train_segmentations_dir, val_images_dir, val_segmentations_dir, save_path, params, device):
    num_epochs = params['num_epochs']  
    learning_rate = params['learning_rate']
    power = params['power']
    batch_size = params['batch_size']

    transform = MultiChannelMaskTransform(
        A.Compose([
            A.GaussianBlur(p=0.3),    
            A.Normalize(mean=(0.0,), std=(1.0,)),  
            ToTensorV2()
        ])
    )

    train_dataset = KitsDataset(images_dir=train_images_dir, segmentations_dir=train_segmentations_dir, transform=transform)
    val_dataset = KitsDataset(images_dir=val_images_dir, segmentations_dir=val_segmentations_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResidualUNet(in_channels=1, num_classes=configs.NUM_CLASSES).to(device) 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize PolyLRScheduler
    scheduler = PolyLRScheduler(optimizer, max_epochs=num_epochs, power=power)

    best_val_loss = float('inf')        
    
    metrics_file_path = os.path.join(save_path, 'training_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write(f"{'Epoch':<8} | {'Time (s)':<10} | {'Train Loss':<12} | {'Train Dice':<12} | {'Train IoU':<10} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<10}\n")
        f.write("-" * 140 + "\n") 
        f.flush()
        for epoch in tqdm(range(num_epochs)):
            start_time = time.time()  # Start time for the epoch

            train_loss, train_acc, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
            
            epoch_time = time.time() - start_time  # Calculate the epoch time

            # Step the scheduler
            scheduler.step()

            # Log the metrics to the text file
            f.write(f"{epoch+1:<8} | {epoch_time:<10.2f} | {train_loss:<12.4f} | {train_dice:<12.4f} | {train_iou:<10.4f} | {val_loss:<10.4f} | {val_dice:<10.4f} | {val_iou:<10.4f}\n")
            f.flush()
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f} sec")
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_{epoch}.pth'))
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print("Yay! Best model saved!")
    
    # delete the created fold data folder
    shutil.rmtree(os.path.join(configs.base_processed_path_dir, fold))
    print(f"{fold} data deleted successfully!")


if __name__ == '__main__':
    fold = 'fold_4'
    kind = 'train'
    device = torch.device("cuda:1")
    print(device)
    train_images_dir, train_segmentations_dir, val_images_dir, val_segmentations_dir = create_fold_data(fold, configs.base_processed_path_dir, kind)
    save_path = configs.base_analysis_result_dir
    save_path = os.path.join(save_path, fold, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))  
    os.makedirs(save_path, exist_ok=True)

    params = {
        'num_epochs': 60,
        'learning_rate': 0.0001,
        'power': 0.9,
        'batch_size': configs.BATCH_SIZE
    }

    train(fold, train_images_dir, train_segmentations_dir, val_images_dir, val_segmentations_dir, save_path, params, device)
