import os.path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform
from AI_ultrasound_segmentation.UltrasoundDataset import constructDatasetFromDataFolders
import time
from Utility.generalCV import *
import scipy
def set_seed(seed_testue=42):
    """Set seed for reproducibility."""
    random.seed(seed_testue)  # Python random module
    np.random.seed(seed_testue)  # Numpy library
    torch.manual_seed(seed_testue)  # Torch

    # if using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_testue)
        torch.cuda.manual_seed_all(seed_testue)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()


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



def evaluate(model, loader, threshold=50):
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
            print(f"batch: {batch_index}/{len(loader)}")
            images, labels,skeletons = images.to(device), labels.to(device),skeletons.to(device)
            outputs = torch.sigmoid(model(images)).cpu().numpy()

            start_computing_metrics=time.time()
            images = unnormalize_tensor(images)
            precision_visible, recall_visible, F1_score_visible,precision_invisible, recall_invisible, F1_score_invisible=compute_metrics(images.cpu().numpy(),
                                                                                                                                          outputs,
                                                                                                                                          labels.cpu().numpy(),
                                                                                                                                          threshold=threshold)
            precision_all,recall_all,F1_score_all=compute_metics_1_class(outputs, labels.cpu().numpy())
            precision_visible_list += precision_visible
            recall_visible_list += recall_visible
            F1_score_visible_list += F1_score_visible
            precision_invisible_list += precision_invisible
            recall_invisible_list += recall_invisible
            F1_score_invisible_list += F1_score_invisible
            precision_all_list += precision_all
            recall_all_list += recall_all
            F1_score_all_list += F1_score_all

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



def compute_metrics_global(metrics_list):
    arr=np.vstack(metrics_list)
    nan_mask = np.isnan(arr).any(axis=1)

    # Invert the mask to select rows without NaN values
    non_nan_rows = arr[~nan_mask]
    return non_nan_rows.mean(0)







import argparse
def main():
    parser = argparse.ArgumentParser(description="Setup training configuration")
    parser.add_argument('--dataset_root_folder', type=str, default="./data/UltraBones100k",
                        help='Root directory for the dataset')
    parser.add_argument('--target_specimen_id', type=int, default=-1, help='for cluster training')
    args=parser.parse_args()
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folders_test = []
    specimens_involved_test = [2, 7,8]
    # Adjust the range as needed
    for specimen_id in specimens_involved_test:
        specimen_folder=os.path.join(args.dataset_root_folder,f"specimen{specimen_id:02d}")
        ultrasound_records_folder=os.path.join(specimen_folder,"ultrasound_records")
        for anatomy in ["fibula","foot","tibia"]:
            anatomy_folder=os.path.join(ultrasound_records_folder,anatomy)
            for record_name in os.listdir(anatomy_folder):
                record_folder=os.path.join(anatomy_folder,record_name)
                data_folders_test.append(record_folder)
        #

    transform_test = TrivialTransform(num_ops=1, image_size=[256, 256], train=False)
    dataset_test = constructDatasetFromDataFolders(data_folders_test, transform_test)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                            prefetch_factor=1, persistent_workers=True)
    print(f"size of dataset,  val: {len(dataset_test)}")

    model = torch.load("models/train_on_1_3_4_5_6_9_10_11_12_13_14/epoch_100.pth",weights_only=False)
    model = model.to(device)

    # =========================================================================================================================
    precision,recall,F1_score= evaluate(model, loader_test, threshold=107)
    result_file_name= "result.npz"
    np.savez(result_file_name, precision=precision, recall=recall, F1_score=F1_score)
    plot_metrics("result.npz")





def plot_metrics(result_file):
    data=np.load(result_file)
    precision, recall, F1_score=data['precision'],data['recall'],data['F1_score']

    x = np.arange(0, 2.1, 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot Precision
    axes[0].plot(x, precision[0], label="whole", color='black')
    axes[0].plot(x, precision[1], label="visible", color='blue')
    axes[0].plot(x, precision[2], label="invislbe", color='orange')
    axes[0].set_xlabel("Distance (mm)",fontsize=16)
    axes[0].set_ylabel("Accuracy",fontsize=16)
    axes[0].legend()
    axes[0].grid(True)

    # Plot Recall
    axes[1].plot(x, recall[0], label="whole", color='black')
    axes[1].plot(x, recall[1], label="visible", color='blue')
    axes[1].plot(x, recall[2], label="invislbe", color='orange')
    axes[1].set_xlabel("Distance (mm)",fontsize=16)
    axes[1].set_ylabel("Completeness",fontsize=16)
    axes[1].legend()
    axes[1].grid(True)

    # Plot F1 Score
    axes[2].plot(x, F1_score[0], label="whole", color='black')
    axes[2].plot(x, F1_score[1], label="visible", color='blue')
    axes[2].plot(x, F1_score[2], label="invislbe", color='orange')
    axes[2].set_xlabel("Distance (mm)",fontsize=16)
    axes[2].set_ylabel("F1 Score",fontsize=16)
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()










if __name__ == '__main__':
    plot_metrics("result.npz")
    main()






