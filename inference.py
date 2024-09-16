import torch
import numpy as np
import os
from utils import dice_coefficient, iou_coefficient, load_model , get_image_info, get_last_checkpoint, map_segmentation_to_segmentation_folder
from utils import save_iou_dice_results_per_case, class_specific_dice_and_iou_calculator, save_segmentation, get_list_of_checkpionts
from uncert_utils import calculate_entropy, save_uncertainty_maps
import nibabel as nib
from tqdm import tqdm
import configs as configs
import pandas as pd
from dataset_loader import KitsDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def run_inference(depth_list, case_names, model, device, output_dir, test_folder, models_list=None):
    dice_list = []
    iou_list = []
    class_dice_list = []
    class_iou_list = []
    images_dir = os.path.join(configs.base_processed_path_dir, test_folder, "images")
    segmentations_dir = os.path.join(configs.base_processed_path_dir, test_folder, "segmentations")
    
    dataset = KitsDataset(images_dir=images_dir, segmentations_dir=segmentations_dir, transform=configs.BASE_TRANSFORM)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    segmentation_images = os.path.join(configs.base_case_dir, test_folder)
    depth_list, case_names, affine_matrix_list = get_image_info(original_segment_dir=segmentation_images)
    
    
    with torch.no_grad():
        if models_list is None:
            model = model.to(device)
            model.eval()
            
            for inx, (images, masks) in tqdm(enumerate(test_loader)):
                where_to_save, new_slice_num = map_segmentation_to_segmentation_folder(depth_list, case_names, inx)
                where_to_save = os.path.join(output_dir, where_to_save)
                
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)

                dice_score = dice_coefficient(outputs, masks).item()
                iou_score = iou_coefficient(outputs, masks).item()
                class_dice_scores, class_iou_scores = class_specific_dice_and_iou_calculator(outputs, masks)
                dice_list.append(dice_score)
                iou_list.append(iou_score)
                class_dice_list.append(class_dice_scores)
                class_iou_list.append(class_iou_scores)
                
                save_segmentation(outputs, where_to_save, new_slice_num) 
            save_iou_dice_results_per_case(case_names, dice_list, iou_list, class_dice_list, class_iou_list, depth_list, output_dir)
        else:
            for inx, (images, masks) in tqdm(enumerate(test_loader)):
                where_to_save, new_slice_num = map_segmentation_to_segmentation_folder(depth_list, case_names, inx)
                where_to_save = os.path.join(output_dir, where_to_save)
                prob_maps = []
                for which, model in enumerate(models_list):
                    model = model.to(device)
                    model.eval()
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    if which == len(outputs)-1:
                        dice_score = dice_coefficient(outputs, masks).item()
                        iou_score = iou_coefficient(outputs, masks).item()
                        class_dice_scores, class_iou_scores = class_specific_dice_and_iou_calculator(outputs, masks)
                        dice_list.append(dice_score)
                        iou_list.append(iou_score)
                        class_dice_list.append(class_dice_scores)
                        class_iou_list.append(class_iou_scores)
                        save_segmentation(outputs, where_to_save, new_slice_num)
                    outputs = torch.sigmoid(outputs)
                    outputs = outputs.cpu().detach().numpy()
                    outputs = outputs.squeeze(0)
                    prob_maps.append(outputs)
                    model = model.to('cpu')
                prob_maps = np.stack(prob_maps, axis=0) 
                uncertainty_map = calculate_entropy(prob_maps)
                save_uncertainty_maps(uncertainty_map, where_to_save, new_slice_num)
                

def main():
    test_folder = configs.test_folder
    last_checkpoint_dir = get_last_checkpoint(configs.base_analysis_result_dir)
    model_path = os.path.join(configs.base_analysis_result_dir, last_checkpoint_dir, 'best_model.pth')
    list_of_checkpoints_dirs = get_list_of_checkpionts(configs.base_analysis_result_dir)
    model_list_paths = [os.path.join(configs.base_analysis_result_dir, checkpoint, 'best_model.pth') for checkpoint in list_of_checkpoints_dirs]
    image_path_dir = os.path.join(configs.base_processed_path_dir, test_folder, 'images')
    segment_path_idr = os.path.join(configs.base_processed_path_dir, test_folder, 'segmentations')
    original_segment_dir = os.path.join(configs.base_case_dir, test_folder)
    final_output_dir = os.path.join(configs.base_inference_dir, test_folder)

    image_paths = []
    segmentation_paths = []    
    for image_file_name in os.listdir(image_path_dir):
        image_paths.append(os.path.join(image_path_dir, image_file_name))
    for segmentation_file_name in os.listdir(segment_path_idr):
        segmentation_paths.append(os.path.join(segment_path_idr, segmentation_file_name))
    depth_list, case_names, affine_matrix_list = get_image_info(original_segment_dir)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    models_list = [load_model(model_path, device) for model_path in model_list_paths]
    run_inference(depth_list, case_names, model, device, final_output_dir, test_folder, models_list)

if __name__ == '__main__':
    main()
