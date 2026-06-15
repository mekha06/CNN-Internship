## Task 4: ResNet18 Transfer Learning

In this task, I implemented transfer learning using a pretrained ResNet18 model for leaf disease classification. The main idea behind this task was to move from a small CNN trained from scratch to a stronger pretrained deep learning model that already has prior visual understanding.

In Task 3, the baseline CNN learned image features directly from the leaf disease dataset. That means the model had to learn basic patterns such as edges, textures, spots, color changes, and disease regions only from the available training images. In Task 4, I used ResNet18, a pretrained convolutional neural network, to reuse features learned from a large-scale image dataset and adapt them to my leaf disease classification problem.

This approach is useful because leaf disease datasets are usually limited in size, and training a deep model completely from scratch may not always produce strong results. Transfer learning helps the model learn faster and usually improves performance because the early and middle layers of ResNet18 already know how to detect general visual patterns.

---

### Objective

The objective of this task was to fine-tune `resnet18` pretrained weights on the leaf disease dataset and compare its performance with the baseline CNN model from Task 3.

The specific goals were:

- Load a pretrained ResNet18 model using PyTorch.
- Replace the original ResNet18 classification layer with a custom classifier suitable for the leaf disease classes.
- Apply a freeze and unfreeze training strategy.
- Train the model on the leaf disease training dataset.
- Validate the model using the validation dataset.
- Save the best-performing model weights as `models/resnet18_best.pth`.
- Compare the validation accuracy of ResNet18 with the baseline CNN from Task 3.

The main focus was not only to improve accuracy but also to understand how transfer learning works in a real computer vision pipeline.

---

### Dataset

The dataset was arranged using the `ImageFolder` structure. This structure is useful because PyTorch can automatically assign class labels based on folder names.

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