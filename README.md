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
# Task 2: Custom Dataset and DataLoader

## 1. Task Overview

Task 2 focused on building the **data loading pipeline** for a PyTorch-based leaf disease classification project.

The main goal was to prepare tomato and chilli leaf image datasets in a format that PyTorch can read, create a custom dataset class named `LeafDiseaseDataset`, build train and validation DataLoaders, verify the batch tensor shape, and document the class distribution.

This task did **not** involve model training yet.  
It focused only on preparing the dataset and confirming that images can be loaded correctly in batches.

---

## 2. Objective of Task 2

The task objective was:

> Implement `LeafDiseaseDataset` and train/validation DataLoaders for tomato/chilli leaf classes.

The expected requirements were:

1. Organize images in ImageFolder-style class directories.
2. Implement a custom PyTorch `Dataset` class.
3. Create train and validation DataLoaders.
4. Visualize one batch of images.
5. Confirm the batch shape as `(N, 3, H, W)`.
6. Add a class imbalance summary in the README.
7. Submit the dataset class, DataLoader script, and sample batch figure.

---

## 3. What Was Done

The work was completed in four major stages:

```text
Stage 1: Dataset preparation
Stage 2: Custom Dataset implementation
Stage 3: DataLoader testing and visualization
Stage 4: Documentation and Git cleanup
```

---

# Stage 1: Dataset Preparation

## 4. Dataset Sources Used

Two datasets were used:

```text
1. Tomato leaf disease dataset
2. Chilli plant disease dataset
```

The tomato dataset was not already split into train and validation folders.

The chilli dataset was already split, but the folder structure needed to be copied and cleaned into the project format.

---

## 5. Final Dataset Structure

The final dataset was organized like this:

```text
leaf-disease-cv/
└── data/
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

This is called an **ImageFolder-style structure**.

In this format:

```text
Folder name = class label
Images inside the folder = samples of that class
```

Example:

```text
data/train/Tomato___healthy/image1.jpg
```

This means the image belongs to the class:

```text
Tomato___healthy
```

---

## 6. Classes Used

The final classes used for Task 2 were:

```text
Chili___healthy
Chili___leaf_curl
Tomato___Late_blight
Tomato___healthy
```

These class folder names became the labels for the classification task.

---

## 7. Class Counts

The final training dataset contained:

```text
Chili___healthy: 80 images
Chili___leaf_curl: 80 images
Tomato___Late_blight: 200 images
Tomato___healthy: 200 images
```

Total training images:

```text
560 images
```

The validation dataset contained:

```text
120 images
```

---

## 8. Class Imbalance Observation

The dataset is slightly imbalanced because the tomato classes have more images than the chilli classes.

```text
Tomato classes: 200 images each
Chilli classes: 80 images each
```

This can affect model training later because the model may see tomato examples more often than chilli examples.

Possible future solutions:

```text
- Data augmentation
- Class weights
- Balanced sampling
- Adding more chilli images
```

---

# Stage 1 Code: `split_dataset.py`

## 9. Purpose of `split_dataset.py`

The file `split_dataset.py` was used to split the tomato dataset into train and validation folders.

The tomato dataset originally looked like this:

```text
D:/data-for-leafdisease/
├── Tomato___Late_blight/
└── Tomato___healthy/
```

But the project required this:

```text
leaf-disease-cv/data/
├── train/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
└── val/
    ├── Tomato___Late_blight/
    └── Tomato___healthy/
```

So `split_dataset.py` copied:

```text
200 images per tomato class → train
50 images per tomato class  → val
```

---

## 10. `split_dataset.py` Code

```python
import os
import random
import shutil


# Original downloaded tomato dataset folder
source_root = r"D:\data-for-leafdisease"

# Project data folder
destination_root = r"D:\cnn-internship\leaf-disease-cv\data"

# Tomato classes used for this task
classes = [
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# Number of images per class
train_count = 200
val_count = 50

# To get the same random split every time
random.seed(42)


for class_name in classes:
    source_dir = os.path.join(source_root, class_name)

    train_dir = os.path.join(destination_root, "train", class_name)
    val_dir = os.path.join(destination_root, "val", class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [
        file for file in os.listdir(source_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]

    for image in train_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(train_dir, image)
        shutil.copy2(source_path, destination_path)

    for image in val_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(val_dir, image)
        shutil.copy2(source_path, destination_path)

    print(f"{class_name} completed")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print("-" * 40)


print("Tomato classes split successfully.")
```

---

## 11. Explanation of `split_dataset.py`

### `import os`

```python
import os
```

Purpose:

`os` is used to work with folders and file paths.

It helps with:

```text
- reading folder contents
- joining paths safely
- creating file paths
- checking folders
```

Example:

```python
os.path.join(source_root, class_name)
```

This creates paths like:

```text
D:/data-for-leafdisease/Tomato___healthy
```

---

### `import random`

```python
import random
```

Purpose:

`random` is used to shuffle image filenames before splitting.

Without shuffling, the first 200 images might come from a fixed order, which may not be a good random sample.

---

### `import shutil`

```python
import shutil
```

Purpose:

`shutil` is used to copy files from one folder to another.

Example:

```python
shutil.copy2(source_path, destination_path)
```

This copies the image file while preserving metadata.

---

### `source_root`

```python
source_root = r"D:\data-for-leafdisease"
```

Purpose:

This is the folder where the original tomato class folders are stored.

It should directly contain:

```text
Tomato___Late_blight
Tomato___healthy
```

---

### `destination_root`

```python
destination_root = r"D:\cnn-internship\leaf-disease-cv\data"
```

Purpose:

This is the project dataset folder where train and validation folders are created.

---

### `classes`

```python
classes = [
    "Tomato___Late_blight",
    "Tomato___healthy"
]
```

Purpose:

This list tells the script which class folders to process.

The loop runs once for each class.

---

### `train_count` and `val_count`

```python
train_count = 200
val_count = 50
```

Purpose:

These values control how many images are copied for each class.

For every class:

```text
200 images go to train
50 images go to val
```

---

### `random.seed(42)`

```python
random.seed(42)
```

Purpose:

This makes the random shuffle reproducible.

That means if the script is run again, the same split can be created.

---

### Main Loop

```python
for class_name in classes:
```

Purpose:

This loop processes each target class one by one.

For example:

```text
First loop  → Tomato___Late_blight
Second loop → Tomato___healthy
```

---

### Creating Source and Destination Paths

```python
source_dir = os.path.join(source_root, class_name)

train_dir = os.path.join(destination_root, "train", class_name)
val_dir = os.path.join(destination_root, "val", class_name)
```

Purpose:

These lines dynamically create paths for each class.

Example for `Tomato___healthy`:

```text
source_dir = D:/data-for-leafdisease/Tomato___healthy
train_dir  = D:/cnn-internship/leaf-disease-cv/data/train/Tomato___healthy
val_dir    = D:/cnn-internship/leaf-disease-cv/data/val/Tomato___healthy
```

---

### Creating Folders

```python
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
```

Purpose:

These lines create train and validation folders if they do not already exist.

`exist_ok=True` prevents errors if the folder is already present.

---

### Collecting Image Files

```python
image_files = [
    file for file in os.listdir(source_dir)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]
```

Purpose:

This collects only image files from the class folder.

It ignores non-image files.

Supported formats:

```text
.jpg
.jpeg
.png
```

---

### Shuffling Images

```python
random.shuffle(image_files)
```

Purpose:

This randomizes the order of images before splitting.

This helps prevent biased splits.

---

### Splitting Images

```python
train_images = image_files[:train_count]
val_images = image_files[train_count:train_count + val_count]
```

Purpose:

This selects:

```text
first 200 shuffled images → train
next 50 shuffled images   → val
```

---

### Copying Images

```python
for image in train_images:
    source_path = os.path.join(source_dir, image)
    destination_path = os.path.join(train_dir, image)
    shutil.copy2(source_path, destination_path)
```

Purpose:

This copies training images into the correct train class folder.

The same logic is used for validation images.

---

# Stage 2: Custom Dataset Implementation

## 12. Purpose of `src/dataset.py`

The file `src/dataset.py` contains the custom PyTorch Dataset class.

It tells PyTorch how to:

```text
- scan class folders
- read images
- assign labels
- convert images into tensors
- return image-label pairs
- create DataLoaders
- count images per class
```

This is the most important file for Task 2.

---

## 13. `src/dataset.py` Code

```python
import os
from collections import Counter
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            folder_name for folder_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder_name))
        ])

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)

            for image_name in os.listdir(class_folder):
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_folder, image_name)
                    label = self.class_to_idx[class_name]

                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(train_dir, val_dir, batch_size=16, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = LeafDiseaseDataset(train_dir, transform=transform)
    val_dataset = LeafDiseaseDataset(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataset, val_dataset, train_loader, val_loader


def get_class_counts(dataset):
    counts = Counter(dataset.labels)

    class_counts = {}
    for class_index, count in counts.items():
        class_name = dataset.classes[class_index]
        class_counts[class_name] = count

    return class_counts
```

---

## 14. Explanation of `src/dataset.py`

## 14.1 Imported Modules

### `os`

```python
import os
```

Purpose:

Used for file and folder operations.

It helps scan folders and create correct file paths.

---

### `Counter`

```python
from collections import Counter
```

Purpose:

Used to count the number of images in each class.

This helps create the class imbalance summary.

---

### `PIL Image`

```python
from PIL import Image
```

Purpose:

Used to open image files.

Example:

```python
Image.open(image_path)
```

The `.convert("RGB")` part ensures every image has 3 channels.

---

### `Dataset` and `DataLoader`

```python
from torch.utils.data import Dataset, DataLoader
```

Purpose:

`Dataset` is the base class for creating a custom PyTorch dataset.

`DataLoader` creates batches from the dataset.

---

### `transforms`

```python
from torchvision import transforms
```

Purpose:

Used to preprocess images before sending them to the model.

In this task, transforms were used to:

```text
- resize images
- convert images to tensors
```

---

## 14.2 `LeafDiseaseDataset` Class

```python
class LeafDiseaseDataset(Dataset):
```

Purpose:

This class defines how PyTorch should read the leaf disease dataset.

It inherits from PyTorch's `Dataset` class.

A custom Dataset class usually needs:

```text
__init__
__len__
__getitem__
```

---

## 14.3 `__init__` Method

```python
def __init__(self, root_dir, transform=None):
```

Purpose:

This method runs when a dataset object is created.

Example:

```python
train_dataset = LeafDiseaseDataset("data/train", transform=transform)
```

Parameters:

```text
root_dir  → path to train or val folder
transform → preprocessing applied to images
```

---

### Storing Root Directory and Transform

```python
self.root_dir = root_dir
self.transform = transform
```

Purpose:

These are stored so they can be used later inside the class.

---

### Reading Class Folders

```python
self.classes = sorted([
    folder_name for folder_name in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, folder_name))
])
```

Purpose:

This reads all class folder names inside `data/train` or `data/val`.

Example output:

```text
['Chili___healthy', 'Chili___leaf_curl', 'Tomato___Late_blight', 'Tomato___healthy']
```

`sorted()` is used to keep class order consistent.

---

### Creating Numeric Labels

```python
self.class_to_idx = {
    class_name: index for index, class_name in enumerate(self.classes)
}
```

Purpose:

Neural networks do not understand text labels directly.

So class names are converted into numeric labels.

Example:

```text
Chili___healthy      → 0
Chili___leaf_curl    → 1
Tomato___Late_blight → 2
Tomato___healthy     → 3
```

---

### Creating Image Path and Label Lists

```python
self.image_paths = []
self.labels = []
```

Purpose:

These lists store:

```text
image_paths → full path of every image
labels      → numeric label for every image
```

Example:

```text
image_paths[0] = data/train/Tomato___healthy/image1.jpg
labels[0]      = 3
```

---

### Scanning Each Class Folder

```python
for class_name in self.classes:
    class_folder = os.path.join(root_dir, class_name)
```

Purpose:

This loop goes through each class folder.

Example:

```text
data/train/Chili___healthy
data/train/Chili___leaf_curl
data/train/Tomato___Late_blight
data/train/Tomato___healthy
```

---

### Reading Images Inside Each Class

```python
for image_name in os.listdir(class_folder):
    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
```

Purpose:

This checks every file inside the class folder and keeps only image files.

---

### Saving Image Path and Label

```python
image_path = os.path.join(class_folder, image_name)
label = self.class_to_idx[class_name]

self.image_paths.append(image_path)
self.labels.append(label)
```

Purpose:

This stores the image path and its corresponding numeric label.

This is what makes PyTorch know which image belongs to which class.

---

## 14.4 `__len__` Method

```python
def __len__(self):
    return len(self.image_paths)
```

Purpose:

This returns the total number of images in the dataset.

Example:

```python
len(train_dataset)
```

Output:

```text
560
```

This means there are 560 training images.

---

## 14.5 `__getitem__` Method

```python
def __getitem__(self, index):
```

Purpose:

This method returns one image and one label at a time.

Example:

```python
image, label = train_dataset[0]
```

---

### Getting Image Path and Label

```python
image_path = self.image_paths[index]
label = self.labels[index]
```

Purpose:

This selects the image path and label at the given index.

---

### Opening Image

```python
image = Image.open(image_path).convert("RGB")
```

Purpose:

This opens the image and converts it into RGB format.

RGB means:

```text
Red
Green
Blue
```

This guarantees that each image has 3 channels.

---

### Applying Transform

```python
if self.transform:
    image = self.transform(image)
```

Purpose:

This applies preprocessing to the image.

In this task, preprocessing includes:

```text
- resizing the image to 224x224
- converting the image to a tensor
```

---

### Returning Image and Label

```python
return image, label
```

Purpose:

This returns one processed image and its numeric label.

The model will later use this pair for training.

---

## 15. Image Tensor Explanation

An image tensor is the numerical representation of an image.

A normal image is made of pixels.

Each pixel has RGB values.

Example:

```text
Pixel = [120, 200, 80]
```

This means:

```text
Red   = 120
Green = 200
Blue  = 80
```

When converted into a PyTorch tensor, the image becomes a numerical array.

For one resized RGB image:

```text
Shape = [3, 224, 224]
```

Meaning:

```text
3   = RGB channels
224 = image height
224 = image width
```

A batch of 16 images becomes:

```text
Shape = [16, 3, 224, 224]
```

---

## 16. `create_dataloaders()` Function

```python
def create_dataloaders(train_dir, val_dir, batch_size=16, num_workers=0):
```

Purpose:

This function creates:

```text
- train dataset
- validation dataset
- train DataLoader
- validation DataLoader
```

---

### Transform Pipeline

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

Purpose:

This defines preprocessing steps.

### `Resize((224, 224))`

Resizes every image to:

```text
224 x 224
```

This is important because CNN models need images with the same size.

### `ToTensor()`

Converts images from PIL format to PyTorch tensor format.

It also changes pixel values from:

```text
0 to 255
```

to:

```text
0.0 to 1.0
```

---

### Creating Dataset Objects

```python
train_dataset = LeafDiseaseDataset(train_dir, transform=transform)
val_dataset = LeafDiseaseDataset(val_dir, transform=transform)
```

Purpose:

These lines create dataset objects for train and validation folders.

---

### Creating Train DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
```

Purpose:

This creates batches from the training dataset.

Settings:

```text
batch_size=16 → loads 16 images per batch
shuffle=True → randomizes training image order
num_workers=0 → safe for Windows
```

---

### Creating Validation DataLoader

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
```

Purpose:

This creates batches from the validation dataset.

Settings:

```text
batch_size=16 → loads 16 images per batch
shuffle=False → keeps validation order stable
num_workers=0 → safe for Windows
```

Validation data is usually not shuffled because we want consistent evaluation.

---

## 17. `get_class_counts()` Function

```python
def get_class_counts(dataset):
```

Purpose:

This function counts how many images are present in each class.

---

### Counting Labels

```python
counts = Counter(dataset.labels)
```

Purpose:

This counts how many times each numeric label appears.

Example:

```text
0 → 80
1 → 80
2 → 200
3 → 200
```

---

### Converting Numeric Labels Back to Class Names

```python
class_counts = {}
for class_index, count in counts.items():
    class_name = dataset.classes[class_index]
    class_counts[class_name] = count
```

Purpose:

This converts numeric labels back into readable class names.

Final output:

```text
Chili___healthy: 80 images
Chili___leaf_curl: 80 images
Tomato___Late_blight: 200 images
Tomato___healthy: 200 images
```

---

# Stage 3: DataLoader Testing

## 18. Purpose of `dataloader_test.py`

The file `dataloader_test.py` is used to test whether `src/dataset.py` works correctly.

It checks:

```text
- whether all classes are detected
- whether labels are assigned correctly
- whether image counts are correct
- whether one batch can be loaded
- whether batch shape is correct
- whether images can be visualized
```

---

## 19. `dataloader_test.py` Code

```python
import os
import matplotlib.pyplot as plt

from src.dataset import create_dataloaders, get_class_counts


train_dir = "data/train"
val_dir = "data/val"

train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    batch_size=16,
    num_workers=0
)

print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)

print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))

print("\nClass imbalance summary:")
class_counts = get_class_counts(train_dataset)

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

images, labels = next(iter(train_loader))

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)

os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    image = images[i].permute(1, 2, 0)
    label = labels[i].item()
    class_name = train_dataset.classes[label]

    ax.imshow(image)
    ax.set_title(class_name, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_batch.png")
plt.show()
```

---

## 20. Explanation of `dataloader_test.py`

### Importing Modules

```python
import os
import matplotlib.pyplot as plt
```

Purpose:

`os` is used to create the output folder.

`matplotlib.pyplot` is used to display and save sample batch images.

---

### Importing Functions from `dataset.py`

```python
from src.dataset import create_dataloaders, get_class_counts
```

Purpose:

This imports the DataLoader creation and class counting functions from `src/dataset.py`.

This keeps the project modular.

---

### Defining Dataset Paths

```python
train_dir = "data/train"
val_dir = "data/val"
```

Purpose:

These paths tell the program where the train and validation folders are located.

---

### Creating DataLoaders

```python
train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    batch_size=16,
    num_workers=0
)
```

Purpose:

This creates both datasets and both DataLoaders.

The returned objects are:

```text
train_dataset → training dataset object
val_dataset   → validation dataset object
train_loader  → batch loader for training images
val_loader    → batch loader for validation images
```

---

### Printing Class Names

```python
print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)
```

Purpose:

This verifies that the class folders were detected correctly.

Output:

```text
Classes: ['Chili___healthy', 'Chili___leaf_curl', 'Tomato___Late_blight', 'Tomato___healthy']
Class to index: {'Chili___healthy': 0, 'Chili___leaf_curl': 1, 'Tomato___Late_blight': 2, 'Tomato___healthy': 3}
```

---

### Printing Dataset Sizes

```python
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
```

Purpose:

This confirms the total number of images loaded.

Output:

```text
Number of training images: 560
Number of validation images: 120
```

---

### Printing Class Imbalance Summary

```python
class_counts = get_class_counts(train_dataset)

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")
```

Purpose:

This shows how many images are present in each class.

Output:

```text
Chili___healthy: 80 images
Chili___leaf_curl: 80 images
Tomato___Late_blight: 200 images
Tomato___healthy: 200 images
```

---

### Loading One Batch

```python
images, labels = next(iter(train_loader))
```

Purpose:

This gets one batch from the training DataLoader.

Since the batch size is 16:

```text
images → 16 image tensors
labels → 16 labels
```

---

### Printing Batch Shape

```python
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
```

Output:

```text
Batch image shape: torch.Size([16, 3, 224, 224])
Batch label shape: torch.Size([16])
```

Meaning:

```text
16  = batch size
3   = RGB channels
224 = image height
224 = image width
```

This confirms the required PyTorch format:

```text
(N, 3, H, W)
```

---

### Creating Output Folder

```python
os.makedirs("outputs", exist_ok=True)
```

Purpose:

This creates an `outputs` folder if it does not already exist.

---

### Creating Image Grid

```python
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
```

Purpose:

This creates a figure with:

```text
2 rows
4 columns
```

So it displays 8 images from the batch.

---

### Converting Tensor Format for Display

```python
image = images[i].permute(1, 2, 0)
```

Purpose:

PyTorch stores images as:

```text
C, H, W
```

Matplotlib expects:

```text
H, W, C
```

So this line changes the image shape from:

```text
3, 224, 224
```

to:

```text
224, 224, 3
```

This is only needed for visualization.

---

### Getting Class Name

```python
label = labels[i].item()
class_name = train_dataset.classes[label]
```

Purpose:

The label is numeric, such as:

```text
0
1
2
3
```

This code converts the label back to its class name.

Example:

```text
2 → Tomato___Late_blight
```

---

### Displaying Images

```python
ax.imshow(image)
ax.set_title(class_name, fontsize=8)
ax.axis("off")
```

Purpose:

This displays each image and shows the class name as the title.

---

### Saving Output Figure

```python
plt.savefig("outputs/sample_batch.png")
```

Purpose:

This saves the sample batch visualization.

Generated output:

```text
outputs/sample_batch.png
```

This file is proof that the DataLoader can successfully load and display images.

---

# 21. Terminal Output

After running:

```bash
python dataloader_test.py
```

The output was:

```text
Classes: ['Chili___healthy', 'Chili___leaf_curl', 'Tomato___Late_blight', 'Tomato___healthy']
Class to index: {'Chili___healthy': 0, 'Chili___leaf_curl': 1, 'Tomato___Late_blight': 2, 'Tomato___healthy': 3}
Number of training images: 560
Number of validation images: 120

Class imbalance summary:
Chili___healthy: 80 images
Chili___leaf_curl: 80 images
Tomato___Late_blight: 200 images
Tomato___healthy: 200 images

Batch image shape: torch.Size([16, 3, 224, 224])
Batch label shape: torch.Size([16])
```

This confirms that:

```text
- All 4 classes were detected
- Labels were assigned correctly
- The train dataset contains 560 images
- The validation dataset contains 120 images
- The DataLoader returns image batches correctly
- The batch shape is correct
```

---

# 22. Why the Batch Shape Matters

The task required confirming the shape:

```text
(N, 3, H, W)
```

The output was:

```text
torch.Size([16, 3, 224, 224])
```

This means:

```text
N = 16 images in the batch
3 = RGB color channels
H = 224 image height
W = 224 image width
```

This is the standard image tensor shape expected by PyTorch CNN models.

---

# 23. Output File

The script generated:

```text
outputs/sample_batch.png
```

This image shows a grid of sample images from the training DataLoader with their class names.

This proves that:

```text
- images are being loaded
- labels are matched with images
- tensor conversion is working
- visualization is working
```

---

# 24. Files Created or Updated in Task 2

```text
split_dataset.py
src/dataset.py
dataloader_test.py
outputs/sample_batch.png
README.md
.gitignore
```

---

# 25. Git and Dataset Handling

Initially, dataset images were accidentally pushed to GitHub.

This was corrected by adding the following to `.gitignore`:

```gitignore
data/
*.zip
venv/
__pycache__/
*.pyc
```

Then the dataset images were removed from Git tracking using:

```bash
git rm -r --cached data
```

This keeps the dataset locally on the computer but removes it from GitHub tracking.

Only code, documentation, and the sample output image should be pushed.

---

# 26. Final Submission Checklist

The following files are required for Task 2 submission:

```text
src/dataset.py
dataloader_test.py
split_dataset.py
outputs/sample_batch.png
README.md
requirements.txt
```

Dataset images should not be uploaded to GitHub.

---

# 27. Final Summary

Task 2 successfully created a complete PyTorch data loading pipeline for the leaf disease classification project.

The dataset was organized into ImageFolder-style train and validation directories. A custom PyTorch Dataset class was implemented to read images, convert them into tensors, and return image-label pairs. Train and validation DataLoaders were created to load images in batches. A sample batch was visualized and saved, and the tensor shape was verified as:

```text
torch.Size([16, 3, 224, 224])
```

This confirms that the dataset is ready for the next stage: building and training a CNN model.

