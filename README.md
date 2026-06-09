# Leaf Disease CV
DAY-1

This repository is part of my internship task for setting up a PyTorch workspace for a Plant Leaf Disease Detector project.

The main goal of this task was to create a clean project structure, install the required computer vision libraries, and verify whether PyTorch is running on GPU or CPU.

Since my current system does not have an NVIDIA GPU, PyTorch is running on CPU. I verified this using a separate environment verification script.

---

## Task 1: PyTorch Environment & GPU Verification

### Objective

Configure a PyTorch environment for the Plant Leaf Disease Detector project and verify the available device.

The task includes:

* Creating a virtual environment
* Installing required libraries
* Creating a proper project structure
* Checking CUDA availability
* Logging the device name
* Adding the verification result to the README
* Submitting a screenshot of the verification output

---

## Project Structure

```bash
leaf-disease-cv/
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── 01_leaf_disease_starter_notebook.ipynb
│
├── screenshots/
│   └── gpu_cpu_verification.png
│
├── src/
│   └── verify_environment.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Why this structure is used

### `data/`

This folder is for storing the plant leaf image dataset. Later, the dataset can be arranged into training and validation folders.

### `src/`

This folder contains the main Python source code. For this task, it contains the environment verification script.

### `models/`

This folder will be used to store trained model files in the future.

### `notebooks/`

This folder is used for experiments and step-by-step model development using Jupyter notebooks.

### `screenshots/`

This folder stores proof of task completion, such as the GPU/CPU verification screenshot.

---

## Libraries Used

### `torch`

PyTorch is the main deep learning library used in this project. It will be used for tensor operations, building neural networks, training models, and handling CPU/GPU devices.

### `torchvision`

Torchvision is useful for computer vision projects. It provides image transformations, dataset loading utilities, and pretrained models that can be used later for leaf disease classification.

### `Pillow`

Pillow is used for opening and processing image files. Since this project is based on plant leaf images, image loading is an important part of the pipeline.

### `matplotlib`

Matplotlib is used for visualizing images, training graphs, accuracy curves, loss curves, and model results.

---

## Environment Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

For Git Bash:

```bash
source venv/Scripts/activate
```

For PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install torch torchvision Pillow matplotlib
```

### 4. Save installed libraries

```bash
python -m pip freeze > requirements.txt
```

---

## GPU/CPU Verification

The environment was verified using the following script:

```bash
python src/verify_environment.py
```

### Verification Output

```txt
PyTorch Environment Verification
--------------------------------
torch version       : 2.12.0+cpu
torchvision version : 0.27.0+cpu
Pillow version      : 12.2.0
matplotlib version  : 3.10.9
CUDA available      : False
Device name         : CPU
Tensor device       : cpu
--------------------------------
Environment verification completed successfully.
```

---

## Verification Note

My system does not have an NVIDIA GPU, so CUDA is not available.

```txt
CUDA available: False
Device name: CPU
Tensor device: cpu
```

This means the environment is correctly set up and PyTorch is running on CPU.

The `+cpu` in the PyTorch and Torchvision versions also confirms that the CPU build is installed.

---

## Screenshot

The verification screenshot is saved inside:

```bash
screenshots/gpu_cpu_verification.png
```

---

## Current Status

Task 1 is completed.

The project now has:

* A clean folder structure
* A virtual environment
* Required libraries installed
* A verification script
* CPU verification output
* Screenshot proof
* Pinned dependencies in `requirements.txt`

---

## What can be improved next

This is only the initial setup stage. In the next steps, the project can be improved by:

* Adding the plant leaf disease dataset
* Creating data loading code using `torchvision.datasets.ImageFolder`
* Adding image transformations
* Building a baseline CNN model
* Training and validating the model
* Saving the trained model inside the `models/` folder
* Adding prediction code for testing new leaf images

---

## Task Completion

This task helped me understand how to set up a PyTorch-based computer vision project from scratch and verify whether the environment is using CPU or GPU.

## Task 2: Custom Dataset and DataLoader

### Objective

The objective of this task was to implement a custom PyTorch Dataset class named `LeafDiseaseDataset` and create train/validation DataLoaders for tomato and chilli leaf disease image classification.

---

### Dataset Organization

The dataset was organized using an ImageFolder-style directory structure. Each class has a separate folder inside the `train` and `val` directories.

```text
data/
├── train/
│   ├── Chili___healthy/
│   ├── Chili___leaf_curl/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
│
└── val/
    ├── Chili___healthy/
    ├── Chili___leaf_curl/
    ├── Tomato___Late_blight/
    └── Tomato___healthy/
```

In this structure, each folder name represents a class label.

---

### Dataset Preparation

The tomato dataset was originally not split into training and validation folders. Therefore, a `split_dataset.py` script was created to split the selected tomato classes into `train` and `val` directories.

The chilli dataset was already split, so the required chilli classes were copied into the project dataset structure manually.

Final classes used:

```text
Chili___healthy
Chili___leaf_curl
Tomato___Late_blight
Tomato___healthy
```

---

### Custom Dataset Implementation

A custom PyTorch Dataset class named `LeafDiseaseDataset` was implemented in:

```text
src/dataset.py
```

The dataset class performs the following tasks:

```text
- Scans class folders
- Stores image paths
- Assigns numeric labels to each class
- Opens images using PIL
- Converts images to RGB format
- Resizes images to 224x224
- Converts images into PyTorch tensors
- Returns each image tensor with its corresponding label
```

---

### DataLoader Implementation

Train and validation DataLoaders were created using `torch.utils.data.DataLoader`.

Configuration used:

```text
Batch size: 16
Train shuffle: True
Validation shuffle: False
num_workers: 0
```

`num_workers=0` was used for better compatibility on Windows.

---

### Batch Shape Verification

The DataLoader was tested using:

```text
dataloader_test.py
```

A sample batch was loaded from the training DataLoader.

Output:

```text
Batch image shape: torch.Size([16, 3, 224, 224])
Batch label shape: torch.Size([16])
```

This confirms the required PyTorch image format:

```text
N = batch size
3 = RGB channels
H = image height
W = image width
```

So the final image batch format is:

```text
(N, 3, H, W)
```

---

### Class Imbalance Summary

Training image count:

```text
Chili___healthy: 80 images
Chili___leaf_curl: 80 images
Tomato___Late_blight: 200 images
Tomato___healthy: 200 images
```

The dataset is slightly imbalanced because the tomato classes contain more images than the chilli classes. This can be handled later during model training using data augmentation, class weights, or balanced sampling.

---

### Output Generated

A sample batch visualization was generated and saved as:

```text
outputs/sample_batch.png
```

This image confirms that the DataLoader can successfully load and display images from the dataset.

---

### Files Added for Task 2

```text
src/dataset.py
dataloader_test.py
split_dataset.py
outputs/sample_batch.png
README.md
```
