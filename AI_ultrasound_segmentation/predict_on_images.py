import os
import random
import argparse
import sys
sys.path.append(os.path.abspath('..'))
from Utility.generalCV import *

import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
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


def main_pure_image(example_image_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "models/train_on_1_3_4_5_6_9_10_11_12_13_14/epoch_100.pth"
    model = torch.load(model_path)
    model = model.to(device)

    for file in os.listdir(example_image_folder):
        img_file = os.path.join(example_image_folder, file)
        img_cv2 = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img = Image.open(img_file).convert('L')
        image = F.to_tensor(img)
        image = F.resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
        images = F.normalize(image, mean=0.17475835978984833, std=0.16475939750671387).unsqueeze(0)
        outputs = model(images.to(device))
        outputs = torch.sigmoid(outputs)
        image = images[0][0]
        pred_labels = (outputs > 0.5)
        pred_label = 255 * pred_labels[0][0].cpu().numpy().astype(np.uint8)
        img_cv2_resized=cv2.resize(img_cv2,(256,256))
        cv2.imshow("pred_label", merge_images_horizontally([img_cv2_resized,overlap_image_with_label(img_cv2_resized, pred_label)]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main_whole_dataset(root_folder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    specimen_ids = range(1,15)
    for specimen_id in specimen_ids:
        specimen_folder=os.path.join(root_folder,f"specimen{specimen_id:02d}")
        model_path=os.path.join(specimen_folder,"pretrained_model","epoch_30.pth")
        model = torch.load(model_path, weights_only=False)
        model = model.to(device)
        specimen_folder = os.path.join(root_folder, f"specimen{specimen_id:02d}")
        US_folder=os.path.join(specimen_folder,"ultrasound_records")
        for anatomy in ["fibula","foot","tibia"]:
            anatomy_folder=os.path.join(US_folder,anatomy)
            for record_name in os.listdir(anatomy_folder):
                record_folder=os.path.join(anatomy_folder,record_name)
                if not os.path.isdir(record_folder):
                    continue
                print(record_folder)
                img_folder=os.path.join(record_folder,"UltrasoundImages")
                label_pred_folder=os.path.join(record_folder,"Labels_pred")
                os.makedirs(label_pred_folder,exist_ok=True)


                print(f"working on {record_folder}")
                sweep_df=pd.read_csv(os.path.join(record_folder,"tracking.csv"))
                for i in range(sweep_df.shape[0]):
                    row=sweep_df.iloc[i]
                    img_file = os.path.join(img_folder, f"{int(row['timestamp'])}.png")
                    img_cv2 = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                    img = Image.open(img_file).convert('L')
                    image = F.to_tensor(img)
                    image = F.resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
                    images = F.normalize(image, mean=0.17475835978984833, std=0.16475939750671387).unsqueeze(0)
                    outputs = model(images.to(device))
                    outputs = torch.sigmoid(outputs)
                    image = images[0][0]
                    pred_labels = (outputs > 0.5)
                    pred_label = 255 * pred_labels[0][0].cpu().numpy().astype(np.uint8)
                    pred_label_resized=cv2.resize(pred_label, img_cv2.shape[::-1])
                    # img_masked=(img_cv2*(pred_label_resized>0))
                    # try:
                    #     img_masked=img_masked[img_masked>0]
                    #     print(np.percentile(img_masked, 99))
                    # except Exception as e:
                    #     print(e)
                    label_pred_path=os.path.join(label_pred_folder,f"{int(row['timestamp'])}_label_pred.png")
                    cv2.imwrite(label_pred_path,pred_label_resized)
                    if i%10==0:
                        cv2.imwrite(label_pred_path.replace(".png","_overlay.png"), merge_images_horizontally([img_cv2,overlap_image_with_label(img_cv2, pred_label_resized)]))

                    ########################################################################################
                    # # for paper
                    # paper_img_path=os.path.join(for_paper_folder,f"{row['timestamp']}.png")
                    # paper_label_path=os.path.join(for_paper_folder,f"{row['timestamp']}_pred.png")
                    # cv2.imwrite(paper_img_path,img_cv2)
                    # cv2.imwrite(paper_label_path,overlap_image_with_label(img_cv2, pred_label_resized))
                    # ########################################################################################
                    #
                    # # cv2.imshow("pred_label", merge_images_horizontally(
                    # #     [img_cv2, overlap_image_with_label(img_cv2, pred_label_resized)]))
                    # # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process ultrasound images.")
    parser.add_argument('--example_image_folder', type=str, default="./example_ultrasound_images",
                        help='Folder containing example images for processing')
    parser.add_argument('--dataset_root_folder', type=str, default="./data/AI_Ultrasound_dataset", help='Root directory for the dataset')
    args = parser.parse_args()

    # main_pure_image(args.example_image_folder)
    main_whole_dataset(args.dataset_root_folder)





