import nibabel as nib
import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import argparse
import configs as configs
from sklearn.model_selection import KFold
import json


# Load NIfTI files
def load_nifti(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def one_hot_encode(mask, num_classes=configs.NUM_CLASSES):
    one_hot_mask = np.zeros((num_classes, mask.shape[0], mask.shape[1]))
    for i in range(num_classes):
        one_hot_mask[i, :, :] = (mask == i).astype(configs.MASK_TYPE)
    return one_hot_mask

def preprocess_image(image, mask, num_classes=configs.NUM_CLASSES):
    target_shape=configs.TARGET_SHAPE
    image = (image - np.mean(image)) / np.std(image)
    image_resized = resize(image, target_shape, mode='constant', anti_aliasing=True)
    mask_resized = resize(mask, target_shape, mode='constant', anti_aliasing=False, order=0)
    
    mask_one_hot = one_hot_encode(mask_resized, num_classes=num_classes)
    
    return image_resized, mask_one_hot

def preprocess_image2(image, mask, num_classes=configs.NUM_CLASSES):
    target_shape = configs.TARGET_SHAPE
    
    # Perform Min-Max scaling normalization
    image_min = np.min(image)
    image_max = np.max(image)
    if image_min == image_max:
        image = np.zeros_like(image) 
    else:
        image = (image - image_min) / (image_max - image_min)
    
    image_resized = resize(image, target_shape, mode='constant', anti_aliasing=True)
    mask_resized = resize(mask, target_shape, mode='constant', anti_aliasing=False, order=0)
    
    mask_one_hot = one_hot_encode(mask_resized, num_classes=num_classes)
    
    return image_resized, mask_one_hot

def preprocess_and_save(data_dir, save_dir, num_classes=configs.NUM_CLASSES, kind='train'):
    data_dir = os.path.join(data_dir, kind)
    if kind.startswith("train"):
        case_names = sorted(os.listdir(data_dir))
        slice_counter = 0
        for case_name in tqdm(case_names):
            case_dir = os.path.join(data_dir, case_name)

            images_dir = os.path.join(save_dir, kind, case_name, 'images')
            segmentations_dir = os.path.join(save_dir, kind, case_name, 'segmentations')

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(segmentations_dir, exist_ok=True)

            image_path = os.path.join(case_dir, 'imaging.nii.gz')
            mask_path = os.path.join(case_dir, 'segmentation.nii.gz')

            image = load_nifti(image_path)
            mask = load_nifti(mask_path)
            
            for i in range(image.shape[0]): 
                img_slice, msk_slice = preprocess_image2(image[i, :, :], mask[i, :, :], num_classes)
                img_slice = img_slice.astype(configs.IMAGE_TYPE)
                msk_slice = msk_slice.astype(configs.MASK_TYPE)

                np.save(os.path.join(images_dir, f'image_{slice_counter:06d}.npy'), img_slice)
                np.save(os.path.join(segmentations_dir, f'segmentation_{slice_counter:06d}.npy'), msk_slice)
                slice_counter += 1

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_names = [f'fold_{i}' for i in range(5)]
        
        final_dict = {}

        for i, (train_index, val_index) in enumerate(kf.split(case_names)):
            fold_name = fold_names[i]
            train_cases = [case_names[i] for i in train_index]
            val_cases = [case_names[i] for i in val_index]
            final_dict[fold_name] = {'train': train_cases, 'val': val_cases}

        with open(os.path.join(save_dir, kind, 'folds_map.json'), 'w') as f:
            json.dump(final_dict, f)

    else:
        images_dir = os.path.join(save_dir, kind, 'images')
        segmentations_dir = os.path.join(save_dir, kind, 'segmentations')

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(segmentations_dir, exist_ok=True)

        slice_counter = 0
        
        for case_name in tqdm(os.listdir(data_dir)):
            case_dir = os.path.join(data_dir, case_name)
            image_path = os.path.join(case_dir, 'imaging.nii.gz')
            mask_path = os.path.join(case_dir, 'segmentation.nii.gz')

            image = load_nifti(image_path)
            mask = load_nifti(mask_path)
            
            for i in range(image.shape[0]): 
                img_slice, msk_slice = preprocess_image2(image[i, :, :], mask[i, :, :], num_classes)
                img_slice = img_slice.astype(configs.IMAGE_TYPE)
                msk_slice = msk_slice.astype(configs.MASK_TYPE)

                np.save(os.path.join(images_dir, f'image_{slice_counter:06d}.npy'), img_slice)
                np.save(os.path.join(segmentations_dir, f'segmentation_{slice_counter:06d}.npy'), msk_slice)

                slice_counter += 1

def main():
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--src_dir', type=str, required=False, help='Path to the dataset directory')
    parser.add_argument('--dest_dir', type=str, required=False, help='Path to the directory to save preprocessed data')
    args = parser.parse_args()
    
    if args.src_dir is None:
        data_dir = os.path.join(configs.base_case_dir)
    else:
        data_dir = args.src_dir
    if args.dest_dir is None:
        save_dir = os.path.join(configs.base_processed_path_dir)
    else:
        save_dir = args.dest_dir
    
    num_classes = configs.NUM_CLASSES
    kind = 'train3'
    preprocess_and_save(data_dir, save_dir, num_classes=num_classes, kind=kind)

    # kind = 'unlabeled3'
    # preprocess_and_save(data_dir, save_dir, num_classes=num_classes, kind=kind)

    # kind = 'valid'
    # preprocess_and_save(data_dir, save_dir, num_classes=num_classes, kind=kind)

    # kind = 'test'
    # preprocess_and_save(data_dir, save_dir, num_classes=num_classes, kind=kind)

if __name__ == "__main__":
    main()
