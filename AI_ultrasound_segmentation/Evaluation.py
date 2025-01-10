import copy

import monai.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.transforms import  ToPILImage
import cv2
import shutil
import os
import random
from torchvision.transforms.functional import InterpolationMode
import segmentation_models_pytorch as smp
from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform,TrivialTransform_UDF
from AI_ultrasound_segmentation.UltrasoundDataset import constructDatasetFromDataFolders,cadaver_ids,cadaver_ids_hip
from AI_ultrasound_segmentation.LossFunctions import Dice_and_Skeleton_Loss,Binary_Segmentation_Loss
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import time
from monai.transforms.utils import distance_transform_edt
from torchvision.transforms.functional import to_pil_image
from Utils.generalCV import *
from AI_ultrasound_segmentation.train_lightning import UltrasoundSegmentationModel
import scipy
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy library
    torch.manual_seed(seed_value)  # Torch

    # if using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()
def compute_dice_score(outputs, labels):
    dice_metric = monai.metrics.DiceMetric(reduction='none')
    with torch.no_grad():
        return dice_metric(y_pred=(outputs > 0.5), y=labels).sum().item()

def compute_chamfer_distance(predictions, labels):

    with torch.no_grad():
        chamfer_distances = []
        hausdorff_distances = []
        hausdorff_distances_95 = []

        scale = (950 + 811) / 256 / 2 * 0.054

        for pred_mask, label_mask in zip(predictions, labels):
            if pred_mask.max()< 0.5 or label_mask.sum() == 0:
                continue


            distance_map = monai.transforms.distance_transform_edt(1 - label_mask)
            pred_indices = torch.where(pred_mask[0] > 0.5)
            distances = distance_map[0][pred_indices] * scale

            chamfer_distance = distances.mean()
            hausdorff_distance = distances.max()
            hausdorff_distance_95 = torch.quantile(distances, 0.95)

            chamfer_distances.append(chamfer_distance)
            hausdorff_distances.append(hausdorff_distance)
            hausdorff_distances_95.append(hausdorff_distance_95)

        total_chamfer_distance = torch.sum(torch.tensor(chamfer_distances))
        total_hausdorff_distance = torch.sum(torch.tensor(hausdorff_distances))
        total_hausdorff_distance_95 = torch.sum(torch.tensor(hausdorff_distances_95))

        return total_chamfer_distance, total_hausdorff_distance, total_hausdorff_distance_95


def compute_metrics_gpu(images, predictions, labels, threshold=50):

    threshold=threshold/255

    with torch.no_grad():
        precision_visible_list = []
        recall_visible_list = []
        F1_score_visible_list = []

        precision_invisible_list = []
        recall_invisible_list = []
        F1_score_invisible_list = []

        scale = (950 + 811) / 256 / 2 * 0.054

        for image,pred_mask, label_mask in zip(images,predictions, labels):
            image=unnormalize_tensor(image)
            if pred_mask.max()< 0.5 or label_mask.sum() == 0:
                continue
            label_visible = ((label_mask > 0) & (image >= threshold)).float()
            label_invisible = ((label_mask > 0) & (image < threshold)).float()
            pred_visible = ((pred_mask > 0.5) & (image >= threshold)).float()
            pred_invisible = ((pred_mask > 0.5) & (image < threshold)).float()

            shortest_distance_map_label_visible = monai.transforms.distance_transform_edt(1 - label_visible)
            shortest_distance_map_label_invisible = monai.transforms.distance_transform_edt(1 - label_invisible)
            shortest_distances_pred_2_gt_visible = shortest_distance_map_label_visible[pred_visible>0] * scale
            shortest_distances_pred_2_gt_invisible = shortest_distance_map_label_invisible[pred_invisible>0] * scale

            precision_visible = torch.tensor(
                [(shortest_distances_pred_2_gt_visible < threshold).sum() / len(shortest_distances_pred_2_gt_visible)
                 for threshold in np.arange(0, 2.1, 0.1)])

            if len(shortest_distances_pred_2_gt_invisible)>0:
                precision_invisible = torch.tensor(
                    [(shortest_distances_pred_2_gt_invisible < threshold).sum() / len(
                        shortest_distances_pred_2_gt_invisible)
                     for threshold in np.arange(0, 2.1, 0.1)])
            else:
                precision_invisible=None



            shortest_distance_map_pred_visible = monai.transforms.distance_transform_edt(1 - 1*(pred_visible>0.5))
            shortest_distance_map_pred_invisible = monai.transforms.distance_transform_edt(1 - 1 * (pred_invisible > 0.5))
            shortest_distances_gt_2_pred_visible = shortest_distance_map_pred_visible[label_visible>0.5] * scale
            shortest_distances_gt_2_pred_invisible = shortest_distance_map_pred_invisible[label_invisible>0.5] * scale


            recall_visible = torch.tensor([(shortest_distances_gt_2_pred_visible < threshold).sum() / len(shortest_distances_gt_2_pred_visible) for
                         threshold in np.arange(0, 2.1, 0.1)])
            if len(shortest_distances_gt_2_pred_invisible)>0:
                recall_invisible = torch.tensor([(shortest_distances_gt_2_pred_invisible < threshold).sum() / len(
                    shortest_distances_gt_2_pred_invisible) for
                                               threshold in np.arange(0, 2.1, 0.1)])
            else:
                recall_invisible=None

            F1_score_visible=2*precision_visible*recall_visible/(precision_visible+recall_visible+1e-10)
            if precision_invisible is not None and recall_invisible is not None:
                F1_score_invisible = 2 * precision_invisible * recall_invisible / (precision_invisible + recall_invisible + 1e-10)
            else:
                F1_score_invisible=None

            precision_visible_list.append(precision_visible)
            recall_visible_list.append(recall_visible)
            F1_score_visible_list.append(F1_score_visible)

            if F1_score_invisible is not None:
                precision_invisible_list.append(precision_invisible)
                recall_invisible_list.append(recall_invisible)
                F1_score_invisible_list.append(F1_score_invisible)


        return precision_visible_list, recall_visible_list, F1_score_visible_list,precision_invisible_list, recall_invisible_list, F1_score_invisible_list,


def compute_metrics(images, predictions, labels, threshold=50):
    threshold = threshold / 255

    with torch.no_grad():
        precision_visible_list = []
        recall_visible_list = []
        F1_score_visible_list = []

        precision_invisible_list = []
        recall_invisible_list = []
        F1_score_invisible_list = []

        scale = (950 + 811) / 256 / 2 * 0.054

        for image, pred_mask, label_mask in zip(images, predictions, labels):
            if pred_mask.max() < 0.5 or label_mask.sum() == 0:
                continue
            label_visible = ((label_mask > 0) & (image >= threshold)).astype(np.uint8)
            label_invisible = ((label_mask > 0) & (image < threshold)).astype(np.uint8)
            pred_visible = ((pred_mask > 0.5) & (image >= threshold)).astype(np.uint8)
            pred_invisible = ((pred_mask > 0.5) & (image < threshold)).astype(np.uint8)

            shortest_distance_map_label_visible = scipy.ndimage.distance_transform_edt(1 - label_visible)
            shortest_distance_map_label_invisible = scipy.ndimage.distance_transform_edt(1 - label_invisible)
            shortest_distances_pred_2_gt_visible = shortest_distance_map_label_visible[pred_visible > 0] * scale
            shortest_distances_pred_2_gt_invisible = shortest_distance_map_label_invisible[pred_invisible > 0] * scale

            precision_visible = np.array(
                [(shortest_distances_pred_2_gt_visible < threshold).sum() / len(shortest_distances_pred_2_gt_visible) for threshold in np.arange(0, 2.1, 0.1)])

            if len(shortest_distances_pred_2_gt_invisible) > 0:
                precision_invisible = np.array(
                    [(shortest_distances_pred_2_gt_invisible < threshold).sum() / len(
                        shortest_distances_pred_2_gt_invisible) for threshold in np.arange(0, 2.1, 0.1)])
            else:
                precision_invisible = None

            shortest_distance_map_pred_visible = scipy.ndimage.distance_transform_edt(1 - 1 * (pred_visible > 0.5))
            shortest_distance_map_pred_invisible = scipy.ndimage.distance_transform_edt(
                1 - 1 * (pred_invisible > 0.5))
            shortest_distances_gt_2_pred_visible = shortest_distance_map_pred_visible[label_visible > 0.5] * scale
            shortest_distances_gt_2_pred_invisible = shortest_distance_map_pred_invisible[label_invisible > 0.5] * scale

            recall_visible = np.array(
                [(shortest_distances_gt_2_pred_visible < threshold).sum() / len(shortest_distances_gt_2_pred_visible)
                 for threshold in np.arange(0, 2.1, 0.1)])
            if len(shortest_distances_gt_2_pred_invisible) > 0:
                recall_invisible = np.array([(shortest_distances_gt_2_pred_invisible < threshold).sum() / len(
                    shortest_distances_gt_2_pred_invisible) for threshold in np.arange(0, 2.1, 0.1)])
            else:
                recall_invisible = None

            F1_score_visible = 2 * precision_visible * recall_visible / (precision_visible + recall_visible + 1e-10)
            if precision_invisible is not None and recall_invisible is not None:
                F1_score_invisible = 2 * precision_invisible * recall_invisible / (precision_invisible + recall_invisible + 1e-10)
            else:
                F1_score_invisible = None

            precision_visible_list.append(precision_visible)
            recall_visible_list.append(recall_visible)
            F1_score_visible_list.append(F1_score_visible)

            if F1_score_invisible is not None:
                precision_invisible_list.append(precision_invisible)
                recall_invisible_list.append(recall_invisible)
                F1_score_invisible_list.append(F1_score_invisible)

        return precision_visible_list, recall_visible_list, F1_score_visible_list, precision_invisible_list, recall_invisible_list, F1_score_invisible_list,


def compute_metics_1_class(predictions, labels):
    with torch.no_grad():
        precision_list = []
        recall_list = []
        F1_score_list = []

        scale = (950 + 811) / 256 / 2 * 0.054

        for pred_mask, label_mask in zip(predictions, labels):
            if pred_mask.max()< 0.5 or label_mask.sum() == 0:
                continue


            shortest_distance_map_label = scipy.ndimage.distance_transform_edt(1 - label_mask)
            shortest_distances_pred_2_gt = shortest_distance_map_label[pred_mask>0.5] * scale
            precision=np.array([(shortest_distances_pred_2_gt < threshold).sum() / len(shortest_distances_pred_2_gt) for threshold in np.arange(0, 2.1, 0.1)])

            shortest_distance_map_pred = scipy.ndimage.distance_transform_edt(1 - 1*(pred_mask>0.5))
            shortest_distances_gt_2_pred = shortest_distance_map_pred[label_mask>0.5] * scale
            recall = np.array([(shortest_distances_gt_2_pred < threshold).sum() / len(shortest_distances_gt_2_pred) for
                         threshold in np.arange(0, 2.1, 0.1)])
            F1_score=2*precision*recall/(precision+recall+1e-10)
            precision_list.append(precision)
            recall_list.append(recall)
            F1_score_list.append(F1_score)


        return precision_list, recall_list, F1_score_list



def validate(model, loader,manual=False,threshold=50):
    device="cuda"
    model.eval()
    with torch.no_grad():
        precision_visible_list=[]
        recall_visible_list=[]
        F1_score_visible_list=[]
        precision_invisible_list=[]
        recall_invisible_list=[]
        F1_score_invisible_list=[]
        precision_all_list = []
        recall_all_list = []
        F1_score_all_list = []
        for batch_index, (img_paths,images, labels,skeletons) in enumerate(loader):
            start=time.time()
            print(f"batch: {batch_index}/{len(loader)}")
            images, labels,skeletons = images.to(device), labels.to(device),skeletons.to(device)
            if manual:
                outputs = load_manual_labels(img_paths)
            else:
                outputs = torch.sigmoid(model(images)).cpu().numpy()

            start_computing_metrics=time.time()
            images = unnormalize_tensor(images)
            precision_visible, recall_visible, F1_score_visible,precision_invisible, recall_invisible, F1_score_invisible=compute_metrics(images.cpu().numpy(),
                                                                                                                                          outputs,
                                                                                                                                          labels.cpu().numpy(),
                                                                                                                                          threshold=threshold)
            precision_all,recall_all,F1_score_all=compute_metics_1_class(outputs, labels.cpu().numpy())
            time_for_computing_metrics=time.time()-start_computing_metrics
            precision_visible_list += precision_visible
            recall_visible_list += recall_visible
            F1_score_visible_list += F1_score_visible
            precision_invisible_list += precision_invisible
            recall_invisible_list += recall_invisible
            F1_score_invisible_list += F1_score_invisible
            precision_all_list += precision_all
            recall_all_list += recall_all
            F1_score_all_list += F1_score_all
            print(f"used time: {time.time()-start}, for metrics: {time_for_computing_metrics}")

        precision_visible = compute_metrics_global(precision_visible_list)
        recall_visible = compute_metrics_global(recall_visible_list)
        F1_score_visible =compute_metrics_global(F1_score_visible_list)
        precision_invisible = compute_metrics_global(precision_invisible_list)
        recall_invisible = compute_metrics_global(recall_invisible_list)
        F1_score_invisible = compute_metrics_global(F1_score_invisible_list)
        precision_all = compute_metrics_global(precision_all_list)
        recall_all = compute_metrics_global(recall_all_list)
        F1_score_all = compute_metrics_global(F1_score_all_list)
        return [precision_all,precision_visible,precision_invisible],[recall_all,recall_visible,recall_invisible],[F1_score_all,F1_score_visible,F1_score_invisible]

def load_manual_labels(img_paths):
    res=[]
    for img_path in img_paths:
        manual_label_path = img_path.replace("UltrasoundImages", "LabelsManual").replace(".png", "-labels.png")
        manual_label = cv2.imread(manual_label_path, cv2.IMREAD_GRAYSCALE)
        _, manual_label = cv2.threshold(manual_label, 254, 255, cv2.THRESH_BINARY)
        manual_label = cv2.resize(manual_label, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
        res.append(manual_label)
    res=np.asarray(res)
    return np.expand_dims(res,1)

def compute_metrics_global(metrics_list):
    arr=np.vstack(metrics_list)
    nan_mask = np.isnan(arr).any(axis=1)

    # Invert the mask to select rows without NaN values
    non_nan_rows = arr[~nan_mask]
    return non_nan_rows.mean(0)

def validate_1_class(model, loader):
    device="cuda"
    print(
        f"{'Chamfer Distance (mm)':<30} | {'Hausdorff Distance (mm)':<30} | {'Hausdorff Distance 95% (mm)':<30}")
    print('-' * 200)
    model.eval()
    with torch.no_grad():
        precision_list=[]
        recall_list=[]
        F1_score_list=[]
        for batch_index, (_,images, labels,skeletons) in enumerate(loader):
            images, labels,skeletons = images.to(device), labels.to(device),skeletons.to(device)
            outputs = model(images)
            precision, recall, F1_score=compute_metics_1_class(torch.sigmoid(outputs),labels)
            precision_list+=precision
            recall_list+=recall
            F1_score_list+=F1_score
        precision=torch.vstack(precision_list).mean(0).cpu().numpy()
        recall=torch.vstack(recall_list).mean(0).cpu().numpy()
        F1_score=torch.vstack(F1_score_list).mean(0).cpu().numpy()
        return precision, recall, F1_score

def validate_visualization(model, loader,manual):
    device="cuda"
    model.eval()
    dice_metric=monai.metrics.DiceMetric(reduction='none')
    dice_2_images={}
    with torch.no_grad():
        for batch_index, (img_paths,images, labels, skeletons) in enumerate(loader):
            images, labels, skeletons = images.to(device), labels.to(device), skeletons.to(device)
            outputs = model(images)
            outputs=torch.sigmoid(outputs)
            dices = dice_metric(y_pred=(outputs > 0.5), y=labels)
            for idx,output in enumerate(outputs):
                pred_label = (output > 0.5).cpu().numpy()
                image = images[idx][0]
                image_opencv = tensor_2_opencv(image)
                label = labels[idx][0]
                label_opencv = tensor_2_opencv(label, -1, -1)
                if manual == False:
                    result_image = resize_image_by_scale(merge_images_horizontally(
                        [image_opencv, overlap_image_with_label(image_opencv, pred_label[0], 0.8),
                         overlap_image_with_label(image_opencv, label_opencv, 0.8)]), 1, 1)
                else:
                    manual_label_path = img_paths[idx].replace("UltrasoundImages", "LabelsManual").replace(".png",
                                                                                                         "-labels.png")
                    manual_label = cv2.imread(manual_label_path, cv2.IMREAD_GRAYSCALE)
                    _, manual_label = cv2.threshold(manual_label, 254, 255, cv2.THRESH_BINARY)
                    manual_label = cv2.resize(manual_label, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
                    result_image = merge_images_vertically([
                        merge_images_horizontally(
                            [image_opencv, overlap_image_with_label(image_opencv, label_opencv, 0.8)]),
                        merge_images_horizontally([overlap_image_with_label(image_opencv, pred_label[0], 0.8),
                                                   overlap_image_with_label(image_opencv, manual_label, 0.8)])
                    ])
                dice_2_images[dices[idx].item()] = result_image






            # cv2.imshow(f"result_{dice}", result_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ground truth
        dice_2_images = dict(sorted(dice_2_images.items(), key=lambda item: item[0]))

        dice_scores=np.asarray(list(dice_2_images.keys()))
        print(np.mean(dice_scores))
        print(np.mean(dice_scores[dice_scores>0.4]))
        for dice_score in dice_2_images.keys():
            cv2.imshow(f"{dice_score}",dice_2_images[dice_score])
            cv2.waitKey(0)
            cv2.destroyAllWindows()






def main_classic(manual=False):
    # plot_metrics("./result.npz", "./result_manual.npz")

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folders_train = []
    data_folders_val = []
    dataset_root_folder = "Z:/AI_Ultrasound_dataset"
    cadavers_involved_val = [2, 7,8]
    cadavers_involved_train = [idx for idx in range(1, 15) if
                               idx not in cadavers_involved_val]  # Adjust the range as needed
    # Adjust the range as needed
    for idx in cadavers_involved_train:
        cadaver_id = cadaver_ids[idx]  # Update according to how cadaver_ids are formatted
        data_folders_train += [f"{dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(1, 15)]
    for idx in cadavers_involved_val:
        cadaver_id = cadaver_ids[idx]  # Update according to how cadaver_ids are formatted
        data_folders_val += [f"{dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(1, 15)]
        #

    transform_val = TrivialTransform(num_ops=1, image_size=[256, 256], train=False)
    dataset_val = constructDatasetFromDataFolders(data_folders_val, transform_val,manual=manual)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                            prefetch_factor=1, persistent_workers=True)
    print(f"size of dataset,  val: {len(dataset_val)}")

    model = torch.load("./models/3thin_(resnet34 FPN)_DICE[1]_BCE[1]_skeleton[0.1]_lr[1e-05] TriAug/epoch_101.pth")
    model = model.to(device)
    #=========================================================================================================================
    if manual:
        compute_manual_label_intensity_distribution(model,loader_val)

    validate_visualization(model,loader_val,manual=manual)
    # =========================================================================================================================
    precision,recall,F1_score= validate(model, loader_val,manual=False,threshold=107)
    result_file_name= "./result.npz"
    np.savez(result_file_name, precision=precision, recall=recall, F1_score=F1_score)
    if manual:
        precision, recall, F1_score = validate(model, loader_val, manual=True, threshold=107)
        result_file_name_manual = "./result_manual.npz"
        np.savez(result_file_name_manual, precision=precision, recall=recall, F1_score=F1_score)
        plot_metrics("./result.npz", "./result_manual.npz")
    else:
        plot_metrics("./result.npz")


def main_classic_hip(manual=False):
    # plot_metrics("./result.npz", "./result_manual.npz")

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folders_train = []
    data_folders_val = []
    dataset_root_folder = "F:/AI_Ultrasound_dataset_Hip"
    cadavers_involved_val = [0]
    cadavers_involved_train = [idx for idx in range(1, 15) if
                               idx not in cadavers_involved_val]  # Adjust the range as needed
    # Adjust the range as needed
    for idx in cadavers_involved_val:
        cadaver_id = cadaver_ids_hip[idx]  # Update according to how cadaver_ids are formatted
        data_folders_val += [f"{dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(13, 16)]
        #

    transform_val = TrivialTransform(num_ops=1, image_size=[256, 256], train=False)
    dataset_val = constructDatasetFromDataFolders(data_folders_val, transform_val,manual=manual,image_only=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                            prefetch_factor=1, persistent_workers=True)
    print(f"size of dataset,  val: {len(dataset_val)}")

    model = torch.load("./models/3thin_(resnet34 FPN)_DICE[1]_BCE[1]_skeleton[0.1]_lr[1e-05] TriAug/epoch_101.pth")
    model = model.to(device)
    #=========================================================================================================================
    if manual:
        compute_manual_label_intensity_distribution(model,loader_val)

    validate_visualization(model,loader_val,manual=manual)
    # =========================================================================================================================
    precision,recall,F1_score= validate(model, loader_val,manual=False,threshold=107)
    result_file_name= "./result.npz"
    np.savez(result_file_name, precision=precision, recall=recall, F1_score=F1_score)
    if manual:
        precision, recall, F1_score = validate(model, loader_val, manual=True, threshold=107)
        result_file_name_manual = "./result_manual.npz"
        np.savez(result_file_name_manual, precision=precision, recall=recall, F1_score=F1_score)
        plot_metrics("./result.npz", "./result_manual.npz")
    else:
        plot_metrics("./result.npz")




def plot_metrics(result_file,result_manual_file=None):
    data=np.load(result_file)
    precision, recall, F1_score=data['precision'],data['recall'],data['F1_score']
    if result_manual_file is not None:
        data = np.load(result_manual_file)
        precision_manual, recall_manual, F1_score_manual = data['precision'], data['recall'], data['F1_score']

    x = np.arange(0, 2.1, 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    # Plot Precision
    axes[0].plot(x, precision[0], label="whole", color='black')
    axes[0].plot(x, precision[1], label="visible", color='blue')
    axes[0].plot(x, precision[2], label="invislbe", color='orange')
    if result_manual_file is not None:
        axes[0].plot(x, precision_manual[0], label="whole (manual)", color='black', linestyle='--')
        axes[0].plot(x, precision_manual[1], label="visible (manual)", color='blue', linestyle='--')
        axes[0].plot(x, precision_manual[2], label="invislbe (manual)", color='orange', linestyle='--')
    axes[0].set_xlabel("Distance (mm)",fontsize=16)
    axes[0].set_ylabel("Precision",fontsize=16)
    axes[0].legend()
    axes[0].grid(True)

    # Plot Recall
    axes[1].plot(x, recall[0], label="Recall", color='black')
    axes[1].plot(x, recall[1], label="Recall", color='blue')
    axes[1].plot(x, recall[2], label="Recall", color='orange')
    if result_manual_file is not None:
        axes[1].plot(x, recall_manual[0], label="whole (manual)", color='black', linestyle='--')
        axes[1].plot(x, recall_manual[1], label="visible (manual)", color='blue', linestyle='--')
        axes[1].plot(x, recall_manual[2], label="invislbe (manual)", color='orange', linestyle='--')
    axes[1].set_xlabel("Distance (mm)",fontsize=16)
    axes[1].set_ylabel("Recall",fontsize=16)
    axes[1].legend()
    axes[1].grid(True)

    # Plot F1 Score
    axes[2].plot(x, F1_score[0], label="F1 Score", color='black')
    axes[2].plot(x, F1_score[1], label="F1 Score", color='blue')
    axes[2].plot(x, F1_score[2], label="F1 Score", color='orange')
    if result_manual_file is not None:
        axes[2].plot(x, F1_score_manual[0], label="whole (manual)", color='black', linestyle='--')
        axes[2].plot(x, F1_score_manual[1], label="visible (manual)", color='blue', linestyle='--')
        axes[2].plot(x, F1_score_manual[2], label="invislbe (manual)", color='orange', linestyle='--')
    axes[2].set_xlabel("Distance (mm)",fontsize=16)
    axes[2].set_ylabel("F1 Score",fontsize=16)
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def compute_manual_label_intensity_distribution(model,loader):
    device = "cuda"
    intensities_manual_label=[]
    intensities_gt_label = []
    intensities_pred_label = []
    with torch.no_grad():
        for batch_index, (img_paths, images, labels, skeletons) in enumerate(loader):
            images, labels, skeletons = images.to(device), labels.to(device), skeletons.to(device)
            outputs=torch.sigmoid(model(images))
            pred_labels=(outputs>0.5)
            for idx, img_path in enumerate(img_paths):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                pred_label=pred_labels[idx][0].cpu().numpy().astype(np.uint8)
                pred_label = cv2.resize(pred_label, img.shape[::-1], cv2.INTER_NEAREST_EXACT)
                label = labels[idx].cpu().numpy()[0]
                label = cv2.resize(label.astype(np.uint8), img.shape[::-1],cv2.INTER_NEAREST_EXACT)
                manual_label_path = img_paths[idx].replace("UltrasoundImages", "LabelsManual").replace(".png",
                                                                                                       "-labels.png")

                manual_label = cv2.imread(manual_label_path, cv2.IMREAD_GRAYSCALE)
                _, manual_label = cv2.threshold(manual_label, 254, 255, cv2.THRESH_BINARY)
                intensities_manual_label.append(img[manual_label>0])
                intensities_gt_label.append(img[label > 0])
                intensities_pred_label.append(img[pred_label > 0])


                row1=merge_images_horizontally([img,overlap_image_with_label_two_classes(img, label,threshold=107)])
                row2=merge_images_horizontally([overlap_image_with_label_two_classes(img,pred_label,threshold=107),
                                                        overlap_image_with_label_two_classes(img, manual_label,threshold=107),
                                                        ])
                result_image = merge_images_vertically([row1,row2])
                result_image=resize_image_by_scale(result_image,-2,-2)
                cv2.imshow("img",result_image)
                plot_intensity_distribution_all(intensities_gt_label[-1],intensities_pred_label[-1],intensities_manual_label[-1])
                # cv2.waitKey(0)
    plot_intensity_distribution_all(np.hstack(intensities_gt_label), np.hstack(intensities_pred_label), np.hstack(intensities_manual_label))
    print(len(intensities_manual_label))
    intensities_manual_label=np.hstack(intensities_manual_label)
    plt.figure(figsize=(8, 6))
    plt.hist(intensities_manual_label, bins=256, edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.title('Distribution of 1D Array', fontsize=16)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Show the plot
    plt.show()


def otsu_threshold_1d(intensity_values):
    # Compute histogram and probabilities of each intensity level
    hist, bin_edges = np.histogram(intensity_values, bins=256, range=(0, 256))
    total = len(intensity_values)

    # Compute probability of each intensity level
    prob = hist / total

    # Cumulative sums and cumulative means
    cumulative_prob = np.cumsum(prob)
    cumulative_mean = np.cumsum(np.arange(256) * prob)

    # Total mean
    total_mean = cumulative_mean[-1]

    # Between-class variance
    between_class_variance = (total_mean * cumulative_prob - cumulative_mean) ** 2 / (
                cumulative_prob * (1 - cumulative_prob) + 1e-10)

    # Find the threshold that maximizes between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    return optimal_threshold

def plot_intensity_distribution_all(intensities_gt_label,intensities_pred_label,intensities_manual_label):
    bin_edges=np.arange(0,257)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    # Plot histograms on separate subplots
    freq,_,_=axes[0].hist(intensities_gt_label, bins=bin_edges, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of intensity (GT)')
    axes[0].set_xlabel('Intensity value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_ylim(0,max(freq)*1.05)

    axes[1].hist(intensities_pred_label, bins=bin_edges, edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of intensity (Pred)')
    axes[1].set_xlabel('Intensity value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_ylim(0, max(freq)*1.05)

    axes[2].hist(intensities_manual_label, bins=bin_edges,edgecolor='black', alpha=0.7)
    axes[2].set_title('Distribution of intensity (Manual)')
    axes[2].set_xlabel('Intensity value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_ylim(0, max(freq)*1.05)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    # print(np.unique(intensities_manual_label))
    threshold_value=otsu_threshold_1d(intensities_gt_label)
    print(f"threshold_value_based_on_Otsu:{threshold_value}")
    plt.show()
    # plt.pause(0.001)






if __name__ == '__main__':
    # plot_metrics("./result.npz", "./result_manual.npz")
    main_classic(manual=False)
    # main_classic_hip(manual=False)
    # main_pure_image()






