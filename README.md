# UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction

This is the repository of the UltraBones100k, which is still under development. It contains:
1. **Dataset Access**: Instructions for downloading the dataset.
2. **Bone Segmentation in Ultrasound Images**: Code and pretrained models for accurate bone segmentation in ultrasound imaging.
3. **3D Reconstruction from Ultrasound Sweeps** Example code for reconstructing 3D data from ultrasound sweeps and measure the distance

In case questions, you can create a Github issue within this repository.

# News
- 30.10.2025: We noticed inconsistencies in some file names, which have now been corrected. We recommend re-downloading the dataset. In addition, we will soon release the original CT and MRI scans to support research on topics such as multimodal fusion. Stay tuned.
- 12.06.2025: We have uploaded complete bone labels (including bone shadow), enabling tasks such as full bone surface segmentation, 3D bone shape completion, and more
- 10.06.2025: The code for 3D reconstruction from ultrasound has been added. The code provides visualization and distance evaluation against the ground-truth CT model
- 27.05.2025: We have uploaded the CT bone models for each specimen. You can find them under the root folder of each specimen, detailed in the following section `Dataset File Structure`
- 21.05.2025: Our manuscript has been accepted by Computers in Biology and Medicine. See the reference section.

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

- **Root Folder**: `AI_Ultrasound_dataset` is the main directory.
- **Specimen Folders**: Each specimen folder (e.g., `specimen01`, `specimen02`) contains data for one specimen.
- **Record Folders**: Each specimen folder has subfolders for each record (e.g., `record01`, `record02`).
- **Labels**: Contains the label files for that record.
- **UltrasoundImages**: Contains the ultrasound image files for that record.

```
XXX:\AI_ULTRASOUND_DATASET
├───specimen01
│   ├───CT_bone_model.stl
│   ├───record01
│   │   ├───UltrasoundImages
│   │   │   └───timestamp.png
│   │   ├───Labels
│   │   │   └───timestamp_label.png
│   │   ├───Labels_full
│   │   │   └───timestamp_label.png
│   │   └───tracking.csv
│   ├───record02
│   │   ├───UltrasoundImages
│   │   │   └───timestamp.png
│   │   ├───Labels
│   │   │   └───timestamp_label.png
│   │   ├───Labels_full
│   │   │   └───timestamp_label.png
│   │   └───tracking.csv
│   ├───record03
│   │   ├───UltrasoundImages
│   │   │   └───timestamp.png
│   │   ├───Labels
│   │   │   └───timestamp_label.png
│   │   ├───Labels_full
│   │   │   └───timestamp_label.png
│   │   └───tracking.csv
│   ⋮
│   ⋮
├───specimen02_F231218
├───specimen03_S231783
⋮
⋮
```
# Tracking data
There are two types of tracking data in `tracking.csv`: original (i.e., x) and optimized (i.e., x_optimized). Both are already temporally synchronized. An example code for 3D reconstruction is availble at 
```
3D reconstruction/3D_reconstruction_from_US.py
```

# Train the Bone Segmentation model 
The training script is located at:
``` AI_ultrasound_segmentation/train_lightning.py ```. Train the model using one NVIDIA V100 for 100 epochs, which typically takes around 10 hours. The training process leverages a ResNet-34 FPN architecture with a combination of DICE and BCE losses, and a learning rate of 1e-05.
By default, we assume the dataset folder located at ```../data/AI_Ultrasound_dataset/```

To train the model, just run `python AI_ultrasound_segmentation/train_lightning.py`


# Pretrained model
The pretrained model (trained on specimens [1,3,4,5,6,9,10,11,12,13,14]) is available at 
```
AI_ultrasound_segmentation/models/train_on_1_3_4_5_6_9_10_11_12_13_14/epoch_100.pth
```


# Evaluation
To quantitatively evaluate a trained model on specimens [2,7,8], run:

```
python AI_ultrasound_segmentation/evaluation.py
```


To qualitatively evaluate a trained model on some example ultrasound images, a notebook is available at:

```
AI_ultrasound_segmentation/segment_example_images.ipynb
```



# 3D Reconstruction from ultrasound

To reconstruct point clouds from ultrasound sweeps and evaluate the results against 3D CT bone model, run the following code (you might need to change the dataset path):
```
3D reconstruction/3D_reconstruction_from_US.py
```

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
