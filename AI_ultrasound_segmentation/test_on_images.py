from PIL import Image
import torchvision.transforms.functional as F
import torch
import os
import random
from torchvision.transforms.functional import InterpolationMode
import argparse
from Utils.generalCV import *
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

    model_path = "./models/3thin_(resnet34 FPN)_DICE[1]_BCE[1]_skeleton[0.1]_lr[1e-05] TriAug/epoch_101.pth"
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
        pred_label = cv2.resize(pred_label, img.size[::], cv2.INTER_NEAREST_EXACT)
        cv2.imshow("pred_label", overlap_image_with_label(img_cv2, pred_label))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process ultrasound images.")
    parser.add_argument('--example_image_folder', type=str, default="./example_ultrasound_images",
                        help='Folder containing example images for processing')
    args = parser.parse_args()

    main_pure_image(args.example_image_folder)





