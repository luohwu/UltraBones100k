import os
import time

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from Utils.generalCV import *
import torch
from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform
cadaver_ids=[
    "cadaver00-F230837",
    "cadaver01_F231091",
    "cadaver02_F231218",
    "cadaver03_S231783",
    "cadaver04_F231091",
    "cadaver05_S232132L",
    "cadaver06_S231987",
    "cadaver07_S232132R",
    "cadaver08_S231989L",
    "cadaver09_S231989R",
    "cadaver10_S232098L",
    "cadaver11_S232110",
    "cadaver12_S240174",
    "cadaver13_S232110L",
    "cadaver14_S240280"
]





class UltrasoundDataset(Dataset):
    def __init__(self, sweep_df, transform=None,image_only=False):
        self.sweep_df = sweep_df
        self.transform = transform
        self.img_only=image_only

    def __len__(self):
        return len(self.sweep_df)

    def __getitem__(self, idx):
        row = self.sweep_df.iloc[idx]
        img = Image.open(row['img_path']).convert('L')
        if self.img_only:
            img, label, skeleton = self.transform(img, img)
            return row['img_path'], img
        else:
            label = Image.open(row['label_path']).convert('L')
            img, label, skeleton = self.transform(img, label)

            return row['img_path'],img, label, skeleton




def constructDataFrameSingleSweep(dataFolder,manual=False,img_only=False):
    if img_only:
        img_folder = os.path.join(dataFolder, 'UltrasoundImages')
        df=pd.read_csv(os.path.join(dataFolder, 'sweepProcessed_full.csv'))
        df['img_path'] = df['timestamp'].apply(lambda x: os.path.join(img_folder, f"{x}.png"))
        return df
    else:
        label_folder = os.path.join(dataFolder, 'Label_partial_gradient')
        img_folder = os.path.join(dataFolder, 'UltrasoundImages')
        timestamps_target = [int(file[:file.find('_')]) for file in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, file))]
        timestamps_target = set(timestamps_target)
        sweep_df_complete = pd.read_csv(os.path.join(dataFolder, 'sweepProcessedAndOptimized.csv'))
        filtered_df = sweep_df_complete[sweep_df_complete['timestamp'].isin(timestamps_target)].copy()

        if manual:
            manual_label_folder = os.path.join(dataFolder, 'LabelsManual')
            if not os.path.isdir(manual_label_folder):
                return None
            timestamps_manual_labels = [int(file[:file.find('-')]) for file in
                                        os.listdir(manual_label_folder) if'-' in file and os.path.isfile(os.path.join(manual_label_folder, file))]
            filtered_df = filtered_df[filtered_df['timestamp'].isin(timestamps_manual_labels)].copy()

        filtered_df['img_path'] = filtered_df['timestamp'].apply(lambda x: os.path.join(img_folder, f"{x}.png"))
        filtered_df['label_path'] = filtered_df['timestamp'].apply(lambda x: os.path.join(label_folder, f"{x}_label_partial_gradient.png"))
        return filtered_df

def constructDataFrameAllSweeps(dataFolders,manual=False,img_only=False):
    df_single_sweep_list = []
    for dataFolder in dataFolders:
        if not os.path.isdir(dataFolder):
            continue
        df = constructDataFrameSingleSweep(dataFolder,manual,img_only=img_only)
        if df is None:
            continue
        df_single_sweep_list.append(df)
    return pd.concat(df_single_sweep_list, ignore_index=True)

def constructDatasetFromDataFolders(dataFolders, transform,manual=False,UDF=False,image_only=False):

    df_all_sweeps = constructDataFrameAllSweeps(dataFolders,manual=manual,img_only=image_only)
    # df_all_sweeps=df_all_sweeps.sample(frac=1).reset_index(drop=True)
    if UDF:
        dataset = UltrasoundDataset_UDF(df_all_sweeps, transform)
    else:
        dataset = UltrasoundDataset(df_all_sweeps, transform,image_only=image_only)
    return dataset


def constructDatasetFromDataFolders_ImageOnly(dataFolders, transform):

    df_all_sweeps = constructDataFrameAllSweeps(dataFolders,manual=False,img_only=True)
    # df_all_sweeps=df_all_sweeps.sample(frac=1).reset_index(drop=True)
    dataset = UltrasoundDataset(df_all_sweeps, transform,image_only=True)
    return dataset


def tensor_2_opencv(img_tensor,mean=-1,std=-1):


    if mean>0:
        img_tensor=img_tensor*std+mean

    img_pil = to_pil_image(img_tensor)
    img_np = np.array(img_pil)
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np




def main_overfit():
    transform = TrivialTransform(num_ops=2,image_size=(256, 256),train=True)
    dataset = UltrasoundDatasetOverfit(img_path='./img.png', label_path='./label.png', transform=transform)
    loader = DataLoader(dataset, batch_size=100)  # Load one item at a time for visualization
    i = 0
    while i < 100:
        for images,labels,skeletons in loader:
            print(i)
            i += 1
            img = images[0]
            img_opencv=tensor_2_opencv(img[0],mean=0.17475835978984833,std=0.16475939750671387)
            label=tensor_2_opencv(labels[0])
            skeleton=tensor_2_opencv(skeletons[0])
            # RGB_image.show()
            cv2.imshow('img', merge_images_horizontally([img_opencv,overlap_image_with_label(img_opencv,label),overlap_image_with_label(img_opencv,skeleton)]))

            # show_with_opencv_overlap(image_test,label_test)
            cv2.waitKey(0)

        # break  # Only show the first set of images

def main():

    dataFolders = [f"Z:/AI_Ultrasound_dataset/cadaver01_F231091/Linear18/record{i:02d}" for i in [1,15]]
    transform = TrivialTransform(num_ops=2,image_size=(256, 256),train=True)
    dataset = constructDatasetFromDataFolders(dataFolders, transform)
    loader = DataLoader(dataset, batch_size=5,shuffle=True)  # Load one item at a time for visualization
    i = 0
    while i < 100:
        for _,images, labels,skeletons in loader:
            img = images[0]
            img_opencv = tensor_2_opencv(img[0], mean=0.17475835978984833, std=0.16475939750671387)
            label = tensor_2_opencv(labels[0])
            skeleton = tensor_2_opencv(skeletons[0])
            # RGB_image.show()
            cv2.imshow('img', merge_images_horizontally([img_opencv, overlap_image_with_label(img_opencv, label),
                                                         overlap_image_with_label(img_opencv, skeleton)]))

            # show_with_opencv_overlap(image_test,label_test)
            cv2.waitKey(0)




def compute_mean_and_std():
    dataFolders = []
    dataset_root_folder = "Z:/AI_Ultrasound_dataset"
    cadavers_involved = list(range(1, 15))  # Adjust the range as needed
    for idx in cadavers_involved:
        cadaver_id = cadaver_ids[idx]  # Update according to how cadaver_ids are formatted
        dataFolders += [f"{dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(1, 15)]

    transform = ConsistentRandomTransform(output_size=(256, 256), enable_vertical_flip=False,
                                          enable_horizontal_flip=False, enable_affine=False,
                                          interpolation=InterpolationMode.BILINEAR)
    dataset = constructDatasetFromDataFolders(dataFolders, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,num_workers=3)  # Load one item at a time for visualization
    mean = 0.0
    std = 0.0
    pos_rate=0
    nb_samples = 0
    start=time.time()
    for images, distance_maps, labels,skeletons in loader:
        B,C,H,W = images.shape  # Batch size (how many samples in this batch)
        images = images.view(B, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        pos_rate+=labels.sum()/(B*C*H*W)
        nb_samples += B
        print(nb_samples)
    mean /= nb_samples
    std /= nb_samples
    pos_rate/=nb_samples
    print(f"mean: {mean.item()}, std: {std.item()}, pos_rate: {pos_rate.item()}, used time: {time.time()-start}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')
    # compute_mean_and_std()
    # main_overfit()
    main()
        # break  # Only show the first set of images
