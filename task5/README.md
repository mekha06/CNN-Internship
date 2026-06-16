# Task 5: Augmentation and Training Stability

## Project Title

Leaf Disease Classification using ResNet18 with Data Augmentation

---

## Objective

The objective of Task 5 was to improve the training stability and generalization of the ResNet18 leaf disease classification model by adding data augmentation techniques.

In this task, I added image transformations such as:

* Random horizontal flip
* Random rotation
* Color jitter

These transformations help the model learn better by showing it slightly different versions of the same training images.

---

## What is Data Augmentation?

Data augmentation is a technique used to artificially create variations of training images.

For example, one leaf image can be changed by:

* flipping it horizontally
* rotating it slightly
* changing brightness
* changing contrast
* changing color intensity

The label of the image remains the same.

Example:

```text
Original image: Chili leaf curl
Augmented image: Rotated Chili leaf curl
Label: Chili leaf curl
```

This helps the model understand that the disease class should not change just because the leaf is rotated, flipped, or captured in different lighting.

---

## Purpose of Augmentation

In real-world conditions, leaf images may be captured in different ways.

A leaf image can have:

* different lighting
* different angles
* different brightness
* different camera quality
* slight rotation
* flipped orientation

Without augmentation, the model may memorize the training images. With augmentation, the model learns more useful disease patterns instead of memorizing exact image positions or lighting.

This helps reduce overfitting and improves generalization on unseen images.

---

## Dataset

The dataset was arranged using the ImageFolder structure.

```text
leaf-disease-cv/
├── data/
│   ├── train/
│   └── val/
```

The dataset contains leaf images from multiple disease classes.

In this task:

```text
Training images: 839
Validation images: 180
```

---

## Folder Structure for Task 5

```text
leaf-disease-cv/
├── data/
│   ├── train/
│   └── val/
├── task5/
│   ├── augmentation.py
│   ├── README.md
│   ├── outputs/
│   │   ├── task5_augmentation_grid.png
│   │   ├── task5_loss_curve.png
│   │   └── task5_accuracy_curve.png
│   └── models/
│       └── task5_resnet18_augmented.pth
```

---

## Augmentation Pipeline

The augmentation was applied only to the training dataset.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
```

The validation dataset was not augmented.

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

Validation data was kept clean so that the model could be tested fairly on unseen images.

---

## Explanation of Augmentation Techniques

### 1. RandomHorizontalFlip

This randomly flips the image horizontally.

It helps the model learn that the disease remains the same even if the leaf direction changes.

### 2. RandomRotation

This randomly rotates the image by a small angle.

It helps the model handle leaves captured from different angles.

### 3. ColorJitter

This randomly changes brightness, contrast, and saturation.

It helps the model handle real-world lighting changes.

---

## Model Used

The model used in this task was ResNet18 with pretrained ImageNet weights.

ResNet18 was used because it has already learned useful image features from a large dataset. The final fully connected layer was replaced according to the number of leaf disease classes in this project.

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
```

Only the final classification layer was trained.

---

## Loss Function and Optimizer

### Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

CrossEntropyLoss was used because this is a multi-class classification problem.

### Optimizer

```python
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

Adam optimizer was used to update the final layer weights during training.

---

## Training Results

The model was trained for 10 epochs.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ----- | ---------: | -------------: | --------------: | ------------------: |
| 1     |     1.2073 |         56.02% |          0.6223 |              86.67% |
| 2     |     0.6689 |         80.93% |          0.4160 |              91.11% |
| 3     |     0.4464 |         89.15% |          0.3214 |              94.44% |
| 4     |     0.3744 |         91.54% |          0.2473 |              94.44% |
| 5     |     0.3231 |         92.25% |          0.2556 |              92.22% |
| 6     |     0.3091 |         91.54% |          0.1839 |              95.56% |
| 7     |     0.2648 |         92.97% |          0.1767 |              95.56% |
| 8     |     0.2397 |         93.44% |          0.1938 |              95.56% |
| 9     |     0.2429 |         93.09% |          0.1414 |              96.67% |
| 10    |     0.2262 |         94.04% |          0.1542 |              97.22% |

---

## Final Result

The final model achieved:

```text
Training Accuracy: 94.04%
Validation Accuracy: 97.22%
Training Loss: 0.2262
Validation Loss: 0.1542
```

The validation accuracy improved steadily and reached 97.22%.

---

## Training Stability Observation

The model showed stable training because:

* training loss decreased from 1.2073 to 0.2262
* validation loss decreased from 0.6223 to 0.1542
* training accuracy improved from 56.02% to 94.04%
* validation accuracy improved from 86.67% to 97.22%

The validation accuracy did not collapse while the training accuracy increased. This shows that the model was not strongly overfitting.

---

## Overfitting Analysis

Overfitting happens when a model performs very well on training data but poorly on validation data.

In this task, the model did not show strong signs of overfitting because both training and validation performances improved.

The final result was:

```text
Train Accuracy: 94.04%
Validation Accuracy: 97.22%
```

The validation accuracy was slightly higher than the training accuracy. This is acceptable because the training images were augmented, making them harder for the model. The validation images were clean and not augmented.

---

## Generated Outputs

The following outputs were generated:

```text
outputs/task5_augmentation_grid.png
outputs/task5_loss_curve.png
outputs/task5_accuracy_curve.png
models/task5_resnet18_augmented.pth
```

### Augmentation Grid

The augmentation grid shows the effect of different transformations:

* original image
* horizontally flipped image
* rotated image
* color jittered image

This confirms that the augmentation pipeline is working correctly.

### Training Curves

The loss and accuracy curves show the model's learning progress across 10 epochs.

---

## How to Run

Run the script from inside the task5 folder:

```bash
python augmentation.py
```

Or from the project root:

```bash
python task5/augmentation.py
```

---

