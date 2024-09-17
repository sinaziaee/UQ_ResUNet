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
from utils import validate_epoch, combined_loss, ema_update, load_model, get_last_checkpoint
from uncert_utils import calculate_entropy
import configs as configs
import warnings

warnings.filterwarnings('ignore')
print("semi_train2.py")
def generate_pseudo_labels(teacher_model, unlabeled_loader, device):
    teacher_model.eval()
    pseudo_labels = []
    uncertainty_maps = []

    with torch.no_grad():
        for images, segmentations in tqdm(unlabeled_loader):
            images = images.to(device)
            outputs = teacher_model(images)
            uncertainty_map = calculate_entropy(torch.sigmoid(outputs).cpu().numpy())
            pseudo_label = (outputs > 0.5).float()
            pseudo_labels.append(pseudo_label.squeeze(0).cpu())
            uncertainty_maps.append(torch.tensor(uncertainty_map).unsqueeze(0))

    return pseudo_labels, uncertainty_maps

# Directories
images_dir = os.path.join(configs.base_processed_path_dir, 'train', 'images')
segmentations_dir = os.path.join(configs.base_processed_path_dir, 'train', 'segmentations')
unlabeled_images_dir = os.path.join(configs.base_processed_path_dir, 'unlabeled', 'images')
unlabeled_segmentation_dir = os.path.join(configs.base_processed_path_dir, 'unlabeled', 'segmentations')
save_path = os.path.join(configs.base_analysis_result_dir, 'semi_supervised', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
os.makedirs(save_path, exist_ok=True)

# Hyperparameters
num_epochs = 50
learning_rate = 1e-4
validation_split = 0.2
power = 0.9

# Transformations
transform = MultiChannelMaskTransform(
    A.Compose([
        A.GaussianBlur(p=0.3),
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2()
    ])
)

# Dataset and DataLoader
labeled_dataset = KitsDataset(images_dir=images_dir, segmentations_dir=segmentations_dir, transform=transform)
train_size = int((1 - validation_split) * len(labeled_dataset))
val_size = len(labeled_dataset) - train_size
train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size])

unlabeled_dataset = KitsDataset(images_dir=unlabeled_images_dir, segmentations_dir=unlabeled_segmentation_dir, transform=transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate pseudo labels and uncertainty maps
# teacher_model = ResidualUNet(in_channels=1, num_classes=4).to(device)
# last_checkpoint_dir = get_last_checkpoint(configs.base_analysis_result_dir)
# model_path = os.path.join(configs.base_analysis_result_dir, last_checkpoint_dir, 'best_model.pth')
model_path = "/home/seyedsina.ziaee/datasets/UQ_ResUNet/results/2024-09-16-22-04-11/checkpoint_25.pth"
teacher_model = load_model(model_path, device)
pseudo_labels, uncertainty_maps = generate_pseudo_labels(teacher_model, unlabeled_loader, device)
print("Pseudo labels generated!")

class CombinedDataset(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset, pseudo_labels):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.labeled_dataset) + len(self.unlabeled_dataset)

    def __getitem__(self, idx):
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset[idx]
        else:
            unlabeled_idx = idx - len(self.labeled_dataset)
            unlabeled_image = self.unlabeled_dataset[unlabeled_idx][0]
            pseudo_label = self.pseudo_labels[unlabeled_idx]
            return unlabeled_image, pseudo_label

# combined_dataset = CombinedDataset(train_dataset, unlabeled_dataset, pseudo_labels)
# train_loader = DataLoader(combined_dataset, batch_size=configs.BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False, num_workers=4)
# Initialize Teacher and Student models
student_model = ResidualUNet(in_channels=1, num_classes=4)
student_model.load_state_dict(teacher_model.state_dict())  # Copy teacher weights to student
# Optimizer and Scheduler
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
scheduler = PolyLRScheduler(optimizer, max_epochs=num_epochs, power=power)
best_val_loss = float('inf')
metrics_file_path = os.path.join(save_path, 'training_metrics.txt')
student_model = student_model.to(device)
# print("number of batches:", len(train_loader))
print("device:", device)
with open(metrics_file_path, 'w') as f:
    f.write(f"{'Epoch':<8} | {'Time (s)':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<10}\n")
    f.write("-" * 120 + "\n")
    f.flush()
    
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        
        # Generate pseudo labels and uncertainty maps at the start of each epoch
        pseudo_labels, uncertainty_maps = generate_pseudo_labels(teacher_model, unlabeled_loader, device)
        print(f"Pseudo labels generated for epoch {epoch + 1}!")

        combined_dataset = CombinedDataset(train_dataset, unlabeled_dataset, pseudo_labels)
        train_loader = DataLoader(combined_dataset, batch_size=configs.BATCH_SIZE, shuffle=True, num_workers=4)

        # Training the student model
        student_model.train()
        running_loss = 0.0
        for i, (images, masks) in tqdm(enumerate(train_loader)):
            # print(len(images), len(masks))
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
        
        # Update teacher model with EMA
        ema_update(teacher_model, student_model)

        # Step the scheduler
        scheduler.step()