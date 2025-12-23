# UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction

This is the repository of the UltraBones100k, which is still under development. It contains:
1. **Dataset Access**: Instructions for downloading the dataset.
2. **Bone Segmentation in Ultrasound Images**: Code and pretrained models for accurate bone segmentation in ultrasound imaging.
3. **3D Reconstruction from Ultrasound Sweeps** Example code for reconstructing 3D data from ultrasound sweeps and measure the distance

In case questions, you can create a Github issue within this repository.

# News

- **21.12.2025:** The dataset has been reorganized according to the intended anatomy during scanning. Pretrained segmentation models (trained using a specimen-wise leave-one-out setup) have been uploaded, along with predicted bone labels for each frame. The 3D reconstruction code has been updated, and point clouds with ground-truth (GT) labels and predicted labels are now available.

# Requirements 
Run the following command to install all the packages listed in the `requirements.txt` file: 
```pip install -r requirements.txt```

The code has been tested on the following setup:

- **Operating System**: Windows 10
- **Python Version**: 3.10.0
- **CUDA Version**: 12.1
- **PyTorch Version**: 2.4.0
- **Processor**: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz
- **GPU**: NVIDIA GeForce RTX 3060 (12GB) and NVIDIA V100

# Docker 
A Dockerfile is provided in the root folder. Alternatively, you can use the Docker image `luohwu123/nirr:latest` directly.


# Dataset downloading
## Lower limbs 
1. Install Azure Storage Explorer [here](https://azure.microsoft.com/en-us/products/storage/storage-explorer).
2. On the main page, select **"Connect to resource"**.
3. Select "Storage account"
4. Select "Connection string", then press Next
5. Paste given URL (including BlobEndpoint=) into "Connection String" field and under  "Display name" write a wished name for the storage. This name is defined only for  you on your local machine and doesn't affect the storage itself
6. In the next page select "Connect"
7. By selecting the storage account you have named in step 6, then selecting "Blob  
containers", you will find the shared drive

The URL: BlobEndpoint=[https://rocs3.blob.core.windows.net/;QueueEndpoint=https://rocs3.queue.core.windows.net/;FileEndpoint=https://rocs3.file.core.windows.net/;TableEndpoint=https://rocs3.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19T23:42:28Z&st=2024-12-19T15:42:28Z&spr=https&sig=KWLVjUi%2BBh2FA%2B6VAfUIUBlgQRz7yaQrduCSSBdVs0g%3D](https://rocs3.blob.core.windows.net/;QueueEndpoint=https://rocs3.queue.core.windows.net/;FileEndpoint=https://rocs3.file.core.windows.net/;TableEndpoint=https://rocs3.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19T23:42:28Z&st=2024-12-19T15:42:28Z&spr=https&sig=KWLVjUi%2BBh2FA%2B6VAfUIUBlgQRz7yaQrduCSSBdVs0g%3D "https://rocs3.blob.core.windows.net/;queueendpoint=https://rocs3.queue.core.windows.net/;fileendpoint=https://rocs3.file.core.windows.net/;tableendpoint=https://rocs3.table.core.windows.net/;sharedaccesssignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19t23:42:28z&st=2024-12-19t15:42:28z&spr=https&sig=kwlvjui%2bbh2fa%2b6vafuiublgqrz7yaqrducssbdvs0g%3d")

## More anatomies
We are currently collecting data for additional anatomies, including the spine and hip bones. Stay tuned for updates!
## Dataset File Structure

The dataset is organized as follows:

- **Root folder**: `UltraBones100k` is the main directory.
- **Specimen folders**: Each specimen folder (e.g., `specimen01`, `specimen02`) contains data for a single specimen.
- **CT_bone_segmentations**: Contains CT-derived bone segmentations, including the fibula, foot, and tibia.
- **ultrasound_records**: Within each specimen folder, this directory contains subfolders for each ultrasound record (e.g., `record01`, `record02`). Records are organized by the targeted anatomy. For example, `**/ultrasound_records/tibia/record05` indicates that `record05` targets the tibia.
- **UltrasoundImages**: Contains the ultrasound image files for the corresponding record.
- **Labels**: Contains partial labels for that record.
- **Labels_full**: Contains full labels for the record (including regions in the acoustic shadow).
- **3D_reconstructions**: Contains 3D point clouds reconstructed from the tracking data, using either ground-truth labels or predicted labels.
- **pretrained_model**: pretrained model using a specimen-wise leave-one-out setup.
- **Labels_pred**: the predicted bone labels using the provided pretrained model.

```
XXX:\UltraBones100k
├───specimen01
│   ├───CT_bone_segmentations
│   │   ├───CT_bone_model_merged.stl (fibula+foot+tibia)
│   │   ├───fibula.stl
│   │   ├───foot.stl
│   │   └───tibia.stl
│   ├───pretrained_model
│   │   └───epoch_30.pth
│   └───ultrasound_records
│       ├───fibula
│       │   ├───record01
│       │   │   ├───tracking.csv
│       │   │   ├───UltrasoundImages
│       │   │   │   └───{timestamp}.png
│       │   │   ├───Labels
│       │   │   │   └───{timestamp}_label.png
│       │   │   ├───Labels_full
│       │   │   │   └───{timestamp}_label.png
│       │   │   ├───Labels_pred
│       │   │   │   └───{timestamp}_label_pred.png
│       │   │   └───3D_reconstructions
│       │   │       └───with_GT_labels
│       │   │       │    ├── reconstruction_pcd.xyz (point-cloud reconstruction using the original tracking data)
│       │   │       │    ├── reconstruction_pcd_filtered.xyz (filtered reconstruction that removes points not from the targeted anatomy; e.g., if `record01` targets the tibia, fibula points are filtered out)
│       │   │       │    ├── reconstruction_pcd_optimizedPose.xyz (point-cloud reconstruction using optimized tracking/pose data)
│       │   │       │    └── reconstruction_pcd_filtered_optimizedPose.xyz (filtered reconstruction using optimized tracking/pose data)
│       │   │       └───with_pred_labels
│       │   │            ⋮
│       │   │            ⋮
│       │   │
│       │   ├───record02
│       │   ⋮
│       │   └───recordxx
│       ├───foot
│       │   ⋮
│       └───tibia
│           ⋮
│    
│   
├───specimen02
├───specimen03
⋮
⋮
└───specimen14
```
# Tracking data
There are two types of tracking data in `tracking.csv`: original (i.e., x) and optimized (i.e., x_optimized). Both are already temporally synchronized. An example code for 3D reconstruction is availble at 
```
3D reconstruction/3D_reconstruction_from_US.py
```

# Train the bone segmentation model

We use a specimen-wise leave-one-out setup. For example, data from specimens 2–14 are used for training, and specimen 1 is used for validation. The training script is located at:

`AI_ultrasound_segmentation/train_lightning.py`

Train the model using one NVIDIA V100 for 100 epochs, which typically takes around 10 hours. The training process uses a ResNet-34 + FPN architecture, optimized with a combination of DICE and BCE losses, and a learning rate of `1e-05`.

By default, the dataset is assumed to be located at:

`../data/UltraBones100k/`

To train the model, run:

`python AI_ultrasound_segmentation/train_lightning.py`

### 🚨 **Recommendation:** The provided script uses a basic segmentation model. More advanced networks (e.g., nnU-Net) and more intensive parameter fine-tuning could likely improve performance.

# Pretrained Models

The pretrained model for each specimen is located at:

`specimenxx/pretrained_model/epoch_30.pth`

Additionally, a pretrained model trained on specimens `[1,3,4,5,6,9,10,11,12,13,14]` is available at:

```text
AI_ultrasound_segmentation/models/train_on_1_3_4_5_6_9_10_11_12_13_14/epoch_100.pth
```
To quantitatively evaluate a trained model on specimens [2,7,8], run:

```
python AI_ultrasound_segmentation/evaluation.py
```


To qualitatively evaluate a trained model on some example ultrasound images, a notebook is available at:

```
AI_ultrasound_segmentation/segment_example_images.ipynb
```



# 3D Reconstruction from ultrasound

To reconstruct point clouds from ultrasound sweeps and evaluate the results against the 3D CT bone model, run the following command (you may need to update the dataset path):

```
python "3D reconstruction/3D_reconstruction_from_US.py" \
  --dataset_root_folder ./data/UltraBones100k \
  --use_pred_label True \
  --use_optimized_pose False
```
Additionally, raw reconstructed point clouds (.xyz) for each record are available under:

- recordxx/3D_reconstructions/with_pred_labels
- recordxx/3D_reconstructions/with_GT_labels

Point clouds reconstructed using predicted bone labels tend to be noisier. Depending on the application, you may apply filtering to remove outliers.


# Reference
```bibtex
@article{WU2025110435,
title = {UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction},
journal = {Computers in Biology and Medicine},
volume = {194},
pages = {110435},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.110435},
url = {https://www.sciencedirect.com/science/article/pii/S0010482525007863},
author = {Luohong Wu and Nicola A. Cavalcanti and Matthias Seibold and Giuseppe Loggia and Lisa Reissner and Jonas Hein and Silvan Beeler and Arnd Viehöfer and Stephan Wirth and Lilian Calvet and Philipp Fürnstahl},
keywords = {Ultrasound bone segmentation, Bone surface segmentation, Bone surface reconstruction, Ultrasound image analysis, Computer-assisted orthopedic surgery},
}
```

# License
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under the Creative Commons Attribution 4.0 International License. Details in license.txt

### Questions or Feedback?

This repository is still under development. If you have questions, you can open a new GitHub issue within this repository, and we'll get back to you!
