import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from Utility.generalCV import *
from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform
specimen_ids=[
    "specimen00",
    "specimen01",
    "specimen02",
    "specimen03",
    "specimen04",
    "specimen05",
    "specimen06",
    "specimen07",
    "specimen08",
    "specimen09",
    "specimen10",
    "specimen11",
    "specimen12",
    "specimen13",
    "specimen14"
]





class UltrasoundDataset(Dataset):
    def __init__(self, sweep_df, transform=None,image_only=False):
        self.sweep_df = sweep_df[:]
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




def constructDataFrameSingleSweep(dataFolder,img_only=False):
    if img_only:
        img_folder = os.path.join(dataFolder, 'UltrasoundImages')
        df=pd.read_csv(os.path.join(dataFolder, 'sweepProcessed_full.csv'))
        df['img_path'] = df['timestamp'].apply(lambda x: os.path.join(img_folder, f"{x}.png"))
        return df
    else:
        label_folder = os.path.join(dataFolder, 'Labels')
        img_folder = os.path.join(dataFolder, 'UltrasoundImages')
        timestamps_target = [int(file[:file.find('_')]) for file in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, file))]
        timestamps_target = set(timestamps_target)
        sweep_df_complete = pd.read_csv(os.path.join(dataFolder, 'tracking.csv'))
        filtered_df = sweep_df_complete[sweep_df_complete['timestamp'].isin(timestamps_target)].copy()
        filtered_df['img_path'] = filtered_df['timestamp'].apply(lambda x: os.path.join(img_folder, f"{x}.png"))
        filtered_df['label_path'] = filtered_df['timestamp'].apply(lambda x: os.path.join(label_folder, f"{x}_label.png"))
        return filtered_df

def constructDataFrameAllSweeps(dataFolders,img_only=False):
    df_single_sweep_list = []
    for dataFolder in dataFolders:
        if not os.path.isdir(dataFolder):
            continue
        df = constructDataFrameSingleSweep(dataFolder,img_only=img_only)
        if df is None:
            continue
        df_single_sweep_list.append(df)
    return pd.concat(df_single_sweep_list, ignore_index=True)

def constructDatasetFromDataFolders(dataFolders, transform,image_only=False):

    df_all_sweeps = constructDataFrameAllSweeps(dataFolders,img_only=image_only)
    dataset = UltrasoundDataset(df_all_sweeps, transform,image_only=image_only)
    return dataset


def tensor_2_opencv(img_tensor,mean=-1,std=-1):


    if mean>0:
        img_tensor=img_tensor*std+mean

    img_pil = to_pil_image(img_tensor)
    img_np = np.array(img_pil)
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np




def main():

    dataFolders = [f"Z:/AI_Ultrasound_dataset/specimen01_F231091/Linear18/record{i:02d}" for i in [1,15]]
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




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')
    # compute_mean_and_std()
    # main_overfit()
    main()
        # break  # Only show the first set of images
