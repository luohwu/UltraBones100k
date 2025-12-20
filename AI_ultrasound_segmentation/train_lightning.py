import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl


import monai
import segmentation_models_pytorch as smp

from AI_ultrasound_segmentation.DataAugmentation import TrivialTransform
from AI_ultrasound_segmentation.UltrasoundDataset import constructDatasetFromDataFolders, specimen_ids
from AI_ultrasound_segmentation.LossFunctions import Binary_Segmentation_Loss
from comet_ml import Experiment
import os
from monai.transforms.utils import distance_transform_edt

import random
import argparse
import numpy as np

pl.seed_everything(42, workers=True)

class UltrasoundSegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-4, DICE_weight=1, BCE_weight=1, skeleton_weight=0.1,val_index=1,comet_experiment=None):
        super(UltrasoundSegmentationModel, self).__init__()
        self.lr = lr
        self.DICE_weight = DICE_weight
        self.BCE_weight = BCE_weight
        self.skeleton_weight = skeleton_weight
        self.dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="sum")
        # Define model
        self.model = smp.FPN(
            encoder_name="resnet34",  # choose encoder
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights
            in_channels=1,  # model input channels (1 for gray-scale images)
            classes=1,  # model output channels (number of classes)
        )
        self.val_index=val_index

        # Define loss function
        self.loss_function = Binary_Segmentation_Loss(DICE_weight=self.DICE_weight, BCE_weight=self.BCE_weight, skeleton_weight=self.skeleton_weight)

        self.comet_experiment=comet_experiment

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

        # Tell Lightning to aggregate this over the epoch (mean by default)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        # Grab the epoch-aggregated value Lightning computed
        train_loss_epoch = self.trainer.callback_metrics.get("train_loss")

        if train_loss_epoch is not None:
            train_loss_epoch = float(train_loss_epoch.detach().cpu())
            print(f"Epoch {self.current_epoch} Train loss (epoch): {train_loss_epoch:.5f}")

            if self.comet_experiment:
                self.comet_experiment.log_metrics(
                    {"train_loss": train_loss_epoch},
                    epoch=self.current_epoch
                )

        # keep your checkpoint saving if you want
        if self.current_epoch % 10 == 0:
            folder = os.path.join("./models", f"validation_on_{self.val_index}")
            os.makedirs(folder, exist_ok=True)
            torch.save(self.model, os.path.join(folder, f"epoch_{self.current_epoch}.pth"))

    def validation_step(self, batch, batch_idx):
        _, images, labels, skeletons = batch
        outputs = self(images)
        loss = self.loss_function(outputs, labels, skeletons)

        dice_score = self.compute_dice_score(torch.sigmoid(outputs), labels)

        chamfer_distance, hausdorff_distance, hausdorff_distance_95 = self.compute_chamfer_distance(
            torch.sigmoid(outputs), labels
        )
        num_samples = len(labels)

        # Log for epoch aggregation
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_dice", dice_score / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_cd", chamfer_distance / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_hd", hausdorff_distance / num_samples, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_hd95", hausdorff_distance_95 / num_samples, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)

    def on_validation_epoch_end(self):
        m = self.trainer.callback_metrics  # contains epoch-aggregated logged metrics

        val_loss = m.get("val_loss")
        val_dice = m.get("val_dice")
        val_cd = m.get("val_cd")
        val_hd = m.get("val_hd")
        val_hd95 = m.get("val_hd95")

        # Some can be None early on; guard it
        def to_float(x):
            return None if x is None else float(x.detach().cpu())

        val_loss_f = to_float(val_loss)
        val_dice_f = to_float(val_dice)
        val_cd_f = to_float(val_cd)
        val_hd_f = to_float(val_hd)
        val_hd95_f = to_float(val_hd95)

        if val_loss_f is not None:
            print(
                f"Epoch {self.current_epoch} Val loss (epoch): {val_loss_f:.5f}, "
                f"DICE: {val_dice_f:.5f}, CD: {val_cd_f:.5f}, HD: {val_hd_f:.5f}, HD95: {val_hd95_f:.5f}"
            )

            if self.comet_experiment:
                self.comet_experiment.log_metrics(
                    {
                        "val_loss": val_loss_f,
                        "val_dice": val_dice_f,
                        "val_cd": val_cd_f,
                        "val_hd": val_hd_f,
                        "val_hd95": val_hd95_f,
                    },
                    epoch=self.current_epoch
                )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class UltrasoundDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root_folder, batch_size=32, num_workers=12, validation_index=1, specimen_ids=range(1,15)):
        super(UltrasoundDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root_folder=dataset_root_folder
        self.specimen_ids=specimen_ids
        self.val_idx=validation_index
    def prepare_data(self):
        # Data preparation logic (if any)
        pass

    def setup(self, stage=None):

        data_folders_train = []
        data_folders_val=[]
        for specimen_id in self.specimen_ids:
            specimen_folder=os.path.join(self.dataset_root_folder,f"specimen{specimen_id:02d}")
            US_record_folder=os.path.join(specimen_folder,"ultrasound_records")
            for anatomy in ["fibula","foot","tibia"]:
                anatomy_folder=os.path.join(US_record_folder,anatomy)
                for child_dir in os.listdir(anatomy_folder):
                    if specimen_id ==self.val_idx:
                        data_folders_val.append(os.path.join(anatomy_folder,child_dir))
                    else:
                        data_folders_train.append(os.path.join(anatomy_folder,child_dir))





        transform_train = TrivialTransform(num_ops=5, image_size=[256,256], train=True)
        transform_val = TrivialTransform(num_ops=0, image_size=[256,256], train=False)

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
    parser.add_argument('--num_epochs', type=int, default=110, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--encoder', type=str, default="resnet34 FPN", help='Encoder type for the model')
    parser.add_argument('--dataset_root_folder', type=str, default="./data/AI_Ultrasound_dataset", help='Root directory for the dataset')
    parser.add_argument('--target_specimen_id', type=int, default=-1, help='for cluster training')
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

    specimen_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for val_id in specimen_ids:
        if args.target_specimen_id>0 and val_id!=args.target_specimen_id:
            continue
        print(f"current validation specimen_idx:{val_id}")
        comet_experiment=Experiment(
                    api_key=os.environ.get("COMET_API_KEY"),
                    project_name="us-segmentation",
                    )
        comet_experiment.set_name(f"fpn_resnet34_val{val_id}")

        model = UltrasoundSegmentationModel(lr=lr, DICE_weight=DICE_weight, BCE_weight=BCE_weight, skeleton_weight=skeleton_weight, val_index=val_id,
                                            comet_experiment=comet_experiment)
        data_module = UltrasoundDataModule(batch_size=batch_size, dataset_root_folder=dataset_root_folder, num_workers=num_workers, validation_index=val_id,
                                           specimen_ids=specimen_ids)

        # Setup checkpointing

        # Initialize the trainer with placeholders commented
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu",
            devices=1,
        )

        # Start training
        trainer.fit(model, datamodule=data_module)
        comet_experiment.end()