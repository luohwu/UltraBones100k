import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import monai
import segmentation_models_pytorch as smp

from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform
from AI_ultrasound_segmentation.UltrasoundDataset import constructDatasetFromDataFolders, cadaver_ids
from AI_ultrasound_segmentation.LossFunctions import Binary_Segmentation_Loss

import numpy as np
import time
import os
import scipy
from monai.transforms.utils import distance_transform_edt

import argparse


# Ensure that compute_dice_score and compute_chamfer_distance are available
# If they are not, you need to include their definitions here

class UltrasoundSegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-4, DICE_weight=1, BCE_weight=1, skeleton_weight=0.1,run_name=None):
        super(UltrasoundSegmentationModel, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.DICE_weight = DICE_weight
        self.BCE_weight = BCE_weight
        self.skeleton_weight = skeleton_weight
        self.dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="sum")
        self.run_name=run_name
        # Define model
        self.model = smp.FPN(
            encoder_name="resnet34",  # choose encoder
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights
            in_channels=1,  # model input channels (1 for gray-scale images)
            classes=1,  # model output channels (number of classes)
        )

        # Define loss function
        self.loss_function = Binary_Segmentation_Loss(DICE_weight=self.DICE_weight, BCE_weight=self.BCE_weight, skeleton_weight=self.skeleton_weight)

    def forward(self, x):
        return self.model(x)

    def compute_dice_score(self, outputs, labels, threshold=0.5, eps=1e-6):
        with torch.no_grad():
            return self.dice_metric(y_pred=(outputs>0.5),y=labels).sum().item()

    def compute_chamfer_distance(self, predictions, labels):

        with torch.no_grad():
            chamfer_distances = []
            hausdorff_distances = []
            hausdorff_distances_95 = []

            scale = (950 + 811) / 256 / 2 * 0.054

            for pred_mask, label_mask in zip(predictions, labels):
                if pred_mask.max()< 0.5 or label_mask.sum() == 0:
                    continue


                distance_map = distance_transform_edt(1 - label_mask)
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

    def training_step(self, batch, batch_idx):
        _, images, labels, skeletons = batch
        outputs = self(images)
        loss = self.loss_function(outputs, labels, skeletons)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # dice_score = self.compute_dice_score(torch.sigmoid(outputs), labels)
        # self.log('train_dice', dice_score / len(labels), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def on_train_epoch_end(self):
        if self.current_epoch%5==0:
            folder=os.path.join("./models",self.run_name)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            torch.save(self.model,os.path.join(folder,f"epoch_{self.current_epoch+1}.pth"))

    def validation_step(self, batch, batch_idx):
        _, images, labels, skeletons = batch
        outputs = self(images)
        loss = self.loss_function(outputs, labels, skeletons)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Compute Dice score
        dice_score = self.compute_dice_score(torch.sigmoid(outputs), labels)
        self.log('val_dice', dice_score / len(labels), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Compute Chamfer Distance and other metrics
        chamfer_distance, hausdorff_distance, hausdorff_distance_95 = self.compute_chamfer_distance(torch.sigmoid(outputs), labels)
        num_samples = len(labels)
        self.log('val_chamfer_distance', chamfer_distance / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_hausdorff_distance', hausdorff_distance / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_hausdorff_distance_95', hausdorff_distance_95 / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class UltrasoundDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root_folder,batch_size=32, num_workers=12,validation_index=1):
        super(UltrasoundDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root_folder=dataset_root_folder
        self.train_all_index=[1,3,4,5,6,9,10,11,12,13,14]
        self.val_idx=validation_index

    def prepare_data(self):
        # Data preparation logic (if any)
        pass

    def setup(self, stage=None):

        cadavers_involved_train = [idx for idx in self.train_all_index if idx!= self.val_idx]
        data_folders_train = []
        for idx in cadavers_involved_train:
            cadaver_id = cadaver_ids[idx]
            data_folders_train += [f"{self.dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(1, 15)]
        data_folders_val = [f"{self.dataset_root_folder}/{cadaver_ids[self.val_idx]}/Linear18/record{i:02d}" for i in range(1, 15)]



        transform_train = TrivialTransform(num_ops=5, image_size=[256,256], train=True)
        transform_val = TrivialTransform(num_ops=1, image_size=[256,256], train=False)

        self.dataset_train = constructDatasetFromDataFolders(data_folders_train, transform_train)
        self.dataset_val = constructDatasetFromDataFolders(data_folders_val, transform_val)

        print(f"Size of dataset - Train: {len(self.dataset_train)}, Validation: {len(self.dataset_val)}")



    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, persistent_workers=True,pin_memory=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False,persistent_workers=True,pin_memory=True,
                          num_workers=self.num_workers)

def parse_args():
    parser = argparse.ArgumentParser(description="Setup training configuration")
    parser.add_argument('--DICE_weight', type=float, default=1, help='Weight for the DICE loss')
    parser.add_argument('--BCE_weight', type=float, default=1, help='Weight for the Binary Cross-Entropy loss')
    parser.add_argument('--skeleton_weight', type=float, default=0.1, help='Weight for the skeleton loss')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--encoder', type=str, default="resnet34 FPN", help='Encoder type for the model')
    parser.add_argument('--dataset_root_folder', type=str, default="Z:/AI_Ultrasound_dataset", help='Root directory for the dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Unpack hyperparameters from args
    DICE_weight = args.DICE_weight
    BCE_weight = args.BCE_weight
    skeleton_weight = args.skeleton_weight
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    encoder = args.encoder
    dataset_root_folder = args.dataset_root_folder

    # Construct the run name dynamically
    run_name = f"3thin_({encoder})_DICE[{DICE_weight}]_BCE[{BCE_weight}]_skeleton[{skeleton_weight}]_lr[{lr}] TriAug"
    print(run_name)
    train_all_index = [1, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
    for val_idx in train_all_index:
        # use multiple GPU to speed up as it can be parallel
        print(f"current validation cadaver_idx:{cadaver_ids[val_idx]}")
        # Placeholder for your model and data module initialization
        model = UltrasoundSegmentationModel(lr=lr, DICE_weight=DICE_weight, BCE_weight=BCE_weight, skeleton_weight=skeleton_weight, run_name=run_name)
        data_module = UltrasoundDataModule(batch_size=batch_size, dataset_root_folder=dataset_root_folder, num_workers=num_workers,validation_index=val_idx)

        # Setup checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join("./models", run_name),
            filename='epoch_{epoch}',
            save_top_k=-1,
            every_n_epochs=5,
        )

        # Initialize the trainer with placeholders commented
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],  # Uncomment as needed
            # profiler="advanced",
            # precision='16-mixed',
            # enable_progress_bar=False
        )

        # Start training
        trainer.fit(model, datamodule=data_module)