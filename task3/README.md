# Day 3 - CNN Architecture and Training

## Overview

Day 3 focused on understanding and implementing a basic Convolutional Neural Network (CNN) for leaf disease image classification using PyTorch.

The main goal was to move from data loading to actual model training by building a small CNN from scratch, training it for multiple epochs, evaluating it on validation data, plotting performance curves, and saving the trained model checkpoint.

---

## Topics Covered

### 1. CNN Basics

A Convolutional Neural Network is mainly used for image-based tasks. CNNs learn useful visual patterns from images such as:

- edges
- spots
- textures
- color changes
- disease marks
- leaf shape patterns

The CNN learns these patterns through convolution filters.

---

### 2. Image Tensor Shape

Images are converted into tensors before being passed to the model.

A single RGB image has the shape:

[3, 128, 128]

This means:

3   -> color channels: Red, Green, Blue  
128 -> image height  
128 -> image width  

When images are loaded in batches, the shape becomes:

[batch_size, channels, height, width]

Example:

[16, 3, 128, 128]

This means 16 images are passed together as one batch.

---

### 3. Image Preprocessing

Before training, all images were resized and converted into tensors using PyTorch transforms.

The preprocessing steps included:

1. Resize all images to 128 x 128
2. Convert images into tensor format

This ensures that all images have the same shape and can be processed by the CNN.

---

### 4. Dataset and DataLoader

The dataset was loaded using PyTorch ImageFolder.

ImageFolder automatically assigns class labels based on folder names.

DataLoader was used to load the images in batches.

Training DataLoader used:

shuffle=True

This helps the model learn better by mixing the image order during training.

Validation DataLoader used:

shuffle=False

Validation only checks model performance, so shuffling is not required.

---

## CNN Architecture Used

A small CNN model was created from scratch using PyTorch.

The model contained:

1. Convolution layers
2. ReLU activation functions
3. MaxPooling layers
4. Flatten layer
5. Fully connected layers

Architecture flow:

Input Image Batch
[16, 3, 128, 128]

↓ Conv2d(3 -> 16)

[16, 16, 128, 128]

↓ MaxPool2d

[16, 16, 64, 64]

↓ Conv2d(16 -> 32)

[16, 32, 64, 64]

↓ MaxPool2d

[16, 32, 32, 32]

↓ Conv2d(32 -> 64)

[16, 64, 32, 32]

↓ MaxPool2d

[16, 64, 16, 16]

↓ Flatten

[16, 16384]

↓ Fully Connected Layers

[16, num_classes]

---

## Important Concepts Learned

### Convolution Layer

A convolution layer applies learnable filters to an image.

Example:

Conv2d(3, 16, kernel_size=3, padding=1)

This means:

3  -> input channels  
16 -> output feature maps  
3  -> 3 x 3 kernel size  
1  -> padding to preserve image size  

The first convolution layer converts an RGB image from 3 channels into 16 feature maps.

---

### Feature Maps

Feature maps are outputs created by convolution filters.

Each filter learns to detect a different visual pattern.

Examples:

- edges
- dark spots
- texture changes
- disease-like patches

Feature maps were also visualized to understand what the first convolution layer produces.

---

### ReLU Activation

ReLU introduces non-linearity into the model.

Formula:

ReLU(x) = max(0, x)

It keeps positive values and converts negative values to zero.

---

### MaxPooling

MaxPooling reduces the height and width of feature maps.

Example:

128 x 128 -> 64 x 64

Pooling helps reduce computation and keeps the strongest features.

---

### Flatten Layer

Flatten converts the feature maps into a single vector before sending them to the fully connected layer.

Example:

[64, 16, 16]

becomes:

64 x 16 x 16 = 16384 features

---

### Fully Connected Layer

Fully connected layers use the extracted CNN features to make the final class prediction.

The final output size is equal to the number of classes.

---

## Training Methodology

1. The image dataset was loaded using PyTorch ImageFolder.

2. Image transformations were applied to resize images and convert them into tensors.

3. DataLoaders were created to load images in batches.

4. A small CNN model was defined using Conv2d, ReLU, MaxPool2d, Flatten, and Linear layers.

5. CrossEntropyLoss was used as the loss function because this is a multi-class classification task.

6. Adam optimizer was used to update the model weights.

7. The model was trained for 3 epochs.

8. During each epoch, the model performed:
   - forward pass
   - loss calculation
   - backward pass
   - optimizer update

9. After each epoch, validation accuracy was calculated.

10. Training loss and validation accuracy were recorded.

11. Loss and accuracy curves were generated.

12. The trained model checkpoint was saved inside the models folder.

---

## Training Loop Workflow

For each epoch:

1. Set model to training mode

model.train()

2. Load one batch of images and labels

3. Move images and labels to the selected device

4. Clear old gradients

optimizer.zero_grad()

5. Perform forward pass

outputs = model(images)

6. Calculate loss

loss = criterion(outputs, labels)

7. Perform backward pass

loss.backward()

8. Update model weights

optimizer.step()

9. Store training loss

10. Evaluate model on validation data

---

## Validation Workflow

During validation:

1. Set model to evaluation mode

model.eval()

2. Disable gradient calculation

with torch.no_grad():

3. Pass validation images through the model

4. Get predicted class using highest output score

5. Compare predictions with actual labels

6. Calculate validation accuracy

Validation accuracy formula:

accuracy = correct predictions / total predictions x 100

---

## Outputs Generated

The following outputs were generated:

1. Trained model checkpoint

models/cnn_leaf_model.pth

2. Training loss curve

outputs/loss_curve.png

3. Validation accuracy curve

outputs/accuracy_curve.png

4. Feature map visualization

outputs/conv1_feature_maps.png

---

## Files Created

train_cnn.py

This file contains:

- imports
- device setup
- image transformations
- dataset loading
- DataLoader creation
- CNN model definition
- feature map visualization
- loss function
- optimizer
- training loop
- validation loop
- model saving
- plot generation

---

## Result Summary

A small CNN was successfully implemented and trained from scratch using PyTorch. The model was trained for 3 epochs using leaf disease images. Training loss and validation accuracy were tracked after each epoch. The final trained model was saved as a checkpoint, and performance curves were generated for analysis.

---

## Key Learnings

Through this task, I learned:

- how CNNs process image data
- how convolution layers create feature maps
- how image dimensions change through Conv2d and MaxPool2d
- how to build a CNN using PyTorch
- how forward pass, loss calculation, backward pass, and optimizer update work
- how to evaluate a model using validation accuracy
- how to save a trained model checkpoint
- how to visualize training performance using loss and accuracy curves
- how to visualize feature maps from convolution layers

---

## Conclusion

Day 3 helped in understanding the complete CNN training pipeline. The task covered both the theory and implementation of a basic CNN model for image classification. By completing this, I gained practical experience in building a deep learning model from scratch, training it, evaluating it, visualizing results, and saving the trained model.
