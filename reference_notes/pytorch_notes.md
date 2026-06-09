PyTorch and TorchVision Notes

What is PyTorch?

PyTorch is an open-source Deep Learning framework developed by Meta (Facebook).

It is used for:

- Building Neural Networks
- Training Machine Learning and Deep Learning models
- Tensor computations
- GPU acceleration for faster processing
- Research and production AI applications

Common Applications

- Image Classification
- Natural Language Processing (NLP)
- Chatbots
- Recommendation Systems
- Object Detection
- Generative AI

Example

import torch

x = torch.tensor([1, 2, 3])
print(x * 2)

Output

tensor([2, 4, 6])

Key Concept: Tensor

A Tensor is the fundamental data structure in PyTorch.

Examples:

Data Type| Example
Scalar| 5
Vector| [1, 2, 3]
Matrix| [[1,2],[3,4]]
Tensor| Multi-dimensional array

---

What is TorchVision?

TorchVision is a library built specifically for Computer Vision tasks using PyTorch.

It provides:

- Image datasets
- Pre-trained models
- Image transformations
- Utilities for loading and processing images

---

Features of TorchVision

1. Datasets

Ready-to-use datasets for training models.

Examples:

- MNIST
- CIFAR-10
- ImageNet

Example:

from torchvision import datasets

mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)

---

2. Pre-trained Models

TorchVision provides pre-trained deep learning models.

Examples:

- ResNet
- VGG
- MobileNet
- EfficientNet

Benefits:

- Saves training time
- Useful for Transfer Learning
- High accuracy on common tasks

---

3. Image Transformations

Used for preprocessing images before training.

Example:

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

Common transformations:

- Resize
- Crop
- Normalize
- Flip
- Rotate

---

PyTorch vs TorchVision

PyTorch| TorchVision
Deep Learning framework| Computer Vision library
Creates and trains models| Handles image-related tasks
Tensor operations| Image datasets and preprocessing
Neural network development| Pre-trained vision models

---

How They Work Together

Images
   ↓
TorchVision
(Load & Preprocess)
   ↓
PyTorch
(Build & Train Model)
   ↓
Predictions

---

Installation

pip install torch torchvision

---

Real-World Projects

PyTorch

- Fake News Detection
- Chatbots
- Sentiment Analysis
- Recommendation Systems

TorchVision

- Cat vs Dog Classifier
- Face Mask Detection
- Plant Disease Detection
- Traffic Sign Recognition
- Object Detection

---

Quick Revision

PyTorch

- Deep Learning Framework
- Neural Networks
- Tensors
- Training Models
- GPU Support

TorchVision

- Computer Vision Toolkit
- Datasets
- Pre-trained Models
- Image Transformations

Easy Memory Trick

PyTorch = Build and Train AI Models

TorchVision = Handle Images for PyTorch

## TASK-2
An image tensor is the numerical form of an image that PyTorch can understand.A computer/model cannot directly understand the picture visually. So the image is converted into numbers.Each pixel has color values.For an RGB image, each pixel has 3 values:
Red value
Green value
Blue value
Example:Pixel = [120, 200, 80]
So the whole image becomes a big collection of numbers.That collection of numbers is called a tensor.

Suppose your image is resized to:224 x 224 and it is an RGB image.Then PyTorch represents one image as:[3, 224, 224]
3   = color channels: Red, Green, Blue
224 = height
224 = width
When DataLoader loads multiple images together, it creates a batch.
If batch size is 16:torch.Size([16, 3, 224, 224])
16  = number of images
3   = RGB channels
224 = height
224 = width

Why do we convert image to tensor?
Because a CNN model can only learn from numbers.

