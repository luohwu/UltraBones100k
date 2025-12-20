import random

import monai.transforms as monai_transforms
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
from scipy.ndimage import distance_transform_edt


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy library

set_seed()  # Call this function at the start of your script


ALL = {
            "identity": v2.Lambda(lambda x: x),
            "horizontal_flip": v2.RandomHorizontalFlip(p=1.0),
            "vertical_flip": v2.RandomVerticalFlip(p=1.0),
            "rotation": v2.RandomRotation(degrees=30),
            "translate_x": v2.RandomAffine(degrees=0, translate=[0.1, 0]),
            "translate_y": v2.RandomAffine(degrees=0, translate=[0, 0.1]),
            "shear_x": v2.RandomAffine(degrees=0, shear=[-30.0, 30.0]),
            "shear_y": v2.RandomAffine(degrees=0, shear=[0.0, 0.0, -30.0, 30.0]),
            "brightness": v2.ColorJitter(brightness=0.5),
            "contrast": v2.ColorJitter(contrast=0.5),
            "saturation": v2.ColorJitter(saturation=0.5),
            "gaussian_blur": v2.GaussianBlur(kernel_size=3),
            "equalize": v2.RandomEqualize(p=1.0),
            "median_blur": monai_transforms.MedianSmooth(radius=3),
            # "scaling": v2.RandomAffine(degrees=0, scale=[0.8, 1.2]),  # Sensible range based on prior works
            # "gaussian_noise": monai_transforms.RandGaussianNoise(prob=1.0),
            # "elastic_transform": v2.ElasticTransform(),
            # "grid_distortion": monai_transforms.RandGridDistortion(prob=1.0),
        }
GEOMETRIC = {
            "identity": v2.Lambda(lambda x: x),
            "horizontal_flip": v2.RandomHorizontalFlip(p=1.0),
            "vertical_flip": v2.RandomVerticalFlip(p=1.0),
            "rotation": v2.RandomRotation(degrees=30),
            "translate_x": v2.RandomAffine(degrees=0, translate=(0.1, 0)),
            "translate_y": v2.RandomAffine(degrees=0, translate=[0, 0.1]),
            "shear_x": v2.RandomAffine(degrees=0, shear=[-30.0, 30.0]),
            # "shear_y": v2.RandomAffine(degrees=0, shear=[0.0, 0.0, -30.0, 30.0]),
            # "scaling": v2.RandomAffine(degrees=0, scale=[0.8, 1.2]),  # Sensible range based on prior works
            # "grid_distortion": monai_transforms.RandGridDistortion(prob=1.0),

            # "elastic_transform": v2.ElasticTransform(),
        }

PHOTOMETRIC = {
            "identity": v2.Lambda(lambda x: x),
            "brightness": v2.ColorJitter(brightness=0.5),
            "contrast": v2.ColorJitter(contrast=0.5),
            "saturation": v2.ColorJitter(saturation=0.5),
            "gaussian_blur": v2.GaussianBlur(kernel_size=3),
            "equalize": v2.RandomEqualize(p=1.0),
            "median_blur": monai_transforms.MedianSmooth(radius=3),
            # "gaussian_noise": monai_transforms.RandGaussianNoise(prob=1.0),
        }


def compute_distance_2Dmap(binary_image, normalized=True, truncated=-1):
    """
    Compute the distance map for a label image where each pixel's value
    is the distance to the closest white (value 255) pixel.

    Args:
    label_image (numpy.ndarray): A label image where white pixels (255) represent areas of interest.

    Returns:
    numpy.ndarray: Distance map where each pixel's value is the distance to the nearest white pixel (255).
    """
    # Ensure the image is boolean where True (1) is 255 and False (0) is the background.

    # Compute the distance transform: distance from non-object (0) pixels to the nearest object (1) pixel
    distance_map = distance_transform_edt(~binary_image)
    if truncated>0:
        distance_map[distance_map > truncated] = truncated
    if normalized:
        distance_map/=distance_map.max()
    # Create the signed distance field where inside mask distances are negative
    return distance_map

class TrivialTransform(torch.nn.Module):
    def __init__(self, num_ops,image_size=[256,256],train=True):
        super().__init__()
        self.num_ops = num_ops
        self.image_size=image_size
        self.mean = 0.17475835978984833
        self.std = 0.16475939750671387
        self.train=train


    def __call__(self, image, label):
        image = F.to_tensor(image)
        image = F.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        label = F.resize(label, self.image_size, interpolation=InterpolationMode.NEAREST_EXACT)

        if self.train:
            # Randomly sample N operations without replacement
            operation_names = random.sample(list(ALL.keys()), self.num_ops)
            # Apply each operation in sequence with the same parameters for both x and label
            for operation_name in operation_names:
                # print(operation_name)
                op = ALL[operation_name]
                if isinstance(op, v2.RandomHorizontalFlip) or isinstance(op, v2.RandomVerticalFlip):
                    image = F.hflip(image) if isinstance(op, v2.RandomHorizontalFlip) else F.vflip(image)
                    label = F.hflip(label) if isinstance(op, v2.RandomHorizontalFlip) else F.vflip(label)
                elif isinstance(op, v2.RandomRotation):
                    angle = op.get_params(op.degrees)
                    image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                    label = F.rotate(label, angle, interpolation=InterpolationMode.NEAREST)
                elif isinstance(op, v2.RandomAffine):
                    params = op.get_params(
                        op.degrees, op.translate, op.scale, op.shear, self.image_size)
                    image = F.affine(image, *params, interpolation=InterpolationMode.BILINEAR)
                    label = F.affine(label, *params, interpolation=InterpolationMode.NEAREST)

                else:
                    # Apply photometric transformations only to the image
                    image = op(image)

        image = F.normalize(image, mean=self.mean, std=self.std)
        distance_map = distance_transform_edt(~(np.asarray(label)))
        label = np.zeros_like(distance_map)
        label[distance_map <= 3] = 1

        skeleton = np.zeros_like(distance_map)
        skeleton[distance_map <= 1] = 1

        return image, F.to_tensor(label),F.to_tensor(skeleton)






