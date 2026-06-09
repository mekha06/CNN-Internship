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
│   └── 01_leaf_disease_starter_notebook.ipynb(need to be created)
│
├── screenshots/
│   └── Screenshot.png
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
python -m pip install torch torchvision Pillow matplotlib scikit-learn tqdm
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
screenshots/Screenshot.png
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
