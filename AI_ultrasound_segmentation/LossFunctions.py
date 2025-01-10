import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
import segmentation_models_pytorch as smp


class Dice_and_Skeleton_Loss(nn.Module):
    def __init__(self,DICE_weight=1,skeleton_weight=1):
        """
        Initialize the WeightedL1Loss with a method choice.

        Args:
            method (str): Method for calculating weights; 'inv_square' for 1/a^2 or 'exp' for exp(-a^2).
        """
        super(Dice_and_Skeleton_Loss, self).__init__()
        self.dice_loss=DiceLoss()
        self.DICE_weight=DICE_weight
        self.skeleton_weight=skeleton_weight

    def forward(self, predictions,target_binary_labels,target_skeletons):
        """
        Calculate the weighted L1 loss where weights are based on the chosen method.

        Args:
            predictions (torch.Tensor): Predicted values.
            target_UDFs (torch.Tensor): Actual GT values.

        Returns:
            torch.Tensor: Computed weighted loss.
        """

        B,C,H,W=predictions.shape
        # weights=1+target_binary_labels
        # weights=torch.ones_like(target_UDFs)
        # weights[target_UDFs<0.9]*=50
        # weights=1



        dice_loss=self.dice_loss(predictions, target_binary_labels)
        #
        # return torch.nn.functional.binary_cross_entropy(soft_binary_label,target_binary_labels)
        skeleton_loss=((predictions*target_skeletons).view(B,-1).sum(1)/target_skeletons.view(B,-1).sum(1)).mean()
        # print(f"L2: {l2_loss}, dice:{dice_loss}, skeleton_loss: {skeleton_loss}")
        return self.DICE_weight*dice_loss-self.skeleton_weight*skeleton_loss



class Binary_Segmentation_Loss(nn.Module):
    def __init__(self, DICE_weight=1,BCE_weight=1,skeleton_weight=1,pos_rate=0.0003015102702192962):
        """
        Initialize the combined loss function with a weighting factor for the SDM product loss.

        Args:
        lambda_val (float): Weight for the product loss component.
        """
        super(Binary_Segmentation_Loss, self).__init__()
        self.dice_loss=smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.dice_loss=DiceLoss()
        self.dice_weight=DICE_weight
        self.pos_rate=pos_rate
        # self.bce_loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1-self.pos_rate)/self.pos_rate]).to("cuda"))
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(81).to("cuda"))
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_weight=BCE_weight
        self.skeleton_weight=skeleton_weight

    def forward(self,predictions,labels,skeletons):
        B, C, H, W = predictions.shape
        skeleton_loss = (
                    (torch.sigmoid(predictions) * skeletons).view(B, -1).sum(1) / skeletons.view(B, -1).sum(1)).mean()
        return self.dice_weight*self.dice_loss(predictions,labels)+self.bce_weight*self.bce_loss(predictions,labels)-self.skeleton_weight*skeleton_loss





# Implementation of ProductLoss as previously defined
class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()

    def forward(self, predictions, targets):
        prod = targets * predictions
        denom = prod + predictions.pow(2) + targets.pow(2)
        denom = torch.clamp(denom, min=1e-8)
        loss_product = -torch.mean(prod / denom)
        return loss_product


# Example usage:
if __name__ == "__main__":
    predictions = torch.tensor([0.5, -0.5, 0.3], dtype=torch.float32)
    targets = torch.tensor([0.5, -0.5, -0.3], dtype=torch.float32)

    # Initialize the combined loss function with lambda_val for the product loss component
    lambda_val = 0.5  # This is a hyperparameter that you can tune
    combined_loss_function = Classic_Semgentaion_Loss(lambda_val=lambda_val)

    loss = combined_loss_function(predictions, targets)
    print("Combined Loss:", loss.item())
