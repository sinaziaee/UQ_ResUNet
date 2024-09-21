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
from utils import train_epoch, validate_epoch, combined_loss, combined_loss_monai, initialize_weights
import configs as configs
import warnings
warnings.filterwarnings('ignore')

def main():
    images_dir = os.path.join(configs.base_processed_path_dir, 'train', 'images')
    segmentations_dir = os.path.join(configs.base_processed_path_dir, 'train', 'segmentations')
    save_path = configs.base_analysis_result_dir
    save_path = os.path.join(save_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))    
    os.makedirs(save_path, exist_ok=True)
    
    num_epochs = 50  # Example: 50 epochs
    learning_rate = 1e-4 # Base learning rate
    validation_split = 0.2
    power = 0.9  # Power for the polynomial decay

    transform = MultiChannelMaskTransform(
        A.Compose([
            A.GaussianBlur(p=0.3),    
            A.Normalize(mean=(0.0,), std=(1.0,)),  
            ToTensorV2()
        ])
    )

    dataset = KitsDataset(images_dir=images_dir, segmentations_dir=segmentations_dir, transform=transform)
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda:1")
    print(device)
    model = ResidualUNet(in_channels=1, num_classes=4).to(device) 
    # model = UNet(in_channels=1, num_classes=4).to(device) 
    # model.apply(initialize_weights)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    criterion = combined_loss
    # criterion = combined_loss_monai2
    # criterion = nn.CrossEntropyLoss()
    # criterion = SoftDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1), batch_dice=True, do_bg=True, smooth=1.0, ddp=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

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
                print("Yay! Best model saved!")


if __name__ == '__main__':
    main()
