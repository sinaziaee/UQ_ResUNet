import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

TARGET_SHAPE = (512, 512)
IMAGE_TYPE = np.float32
MASK_TYPE = np.uint8
BATCH_SIZE = 16
NUM_CLASSES = 4

# base_case_dir = "C:\\datasets\\kits_test\\kits23\\raw"
# base_processed_path_dir = "C:\\datasets\\kits_test\\kits23\\processed"
# base_inference_dir = "C:\\datasets\\kits_test\\kits23\\inference_results"
# base_analysis_result_dir = "C:\\src\\UQ-ResUNet\\results"
# test_folder = "test"

# base_case_dir = "/home/seyedsina.ziaee/datasets/final_kits/raw"
# base_processed_path_dir = "/home/seyedsina.ziaee/datasets/final_kits/processed"
# base_inference_dir = '/home/seyedsina.ziaee/datasets/final_kits/inference_results'
# base_analysis_result_dir = "/home/seyedsina.ziaee/datasets/UQ_ResUNet/results/"
# test_folder = "test3"

base_case_dir = "/scratch/student/sinaziaee/datasets/uq_project/raw"
base_processed_path_dir = "/scratch/student/sinaziaee/datasets/uq_project/processed"
base_inference_dir = '/scratch/student/sinaziaee/datasets/uq_project/inference_results'
base_analysis_result_dir = "/scratch/student/sinaziaee/src/UQ_ResUNet/results/"
test_folder = "test3"


class MultiChannelMaskTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        transformed = self.transform(image=image)
        image = transformed['image']
        return {'image': image, 'mask': torch.tensor(mask, dtype=torch.float32)}

BASE_TRANSFORM = MultiChannelMaskTransform(
        A.Compose([
            A.GaussianBlur(p=0.3),    
            A.Normalize(mean=(0.0,), std=(1.0,)),  
            ToTensorV2()
        ])
    )