import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KitsDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, transform=None):
        self.images_dir = images_dir
        self.segmentations_dir = segmentations_dir
        self.transform = transform

        # List of all image and segmentation filenames
        self.image_filenames = sorted(os.listdir(images_dir))
        self.segmentation_filenames = sorted(os.listdir(segmentations_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image and corresponding segmentation mask
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        msk_path = os.path.join(self.segmentations_dir, self.segmentation_filenames[idx])

        image = np.load(img_path)
        mask = np.load(msk_path)

        # Apply any transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

class CombinedDataset(KitsDataset):
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
            unlabeled_image = self.unlabeled_dataset[unlabeled_idx]
            pseudo_label = self.pseudo_labels[unlabeled_idx]
            return unlabeled_image, pseudo_label