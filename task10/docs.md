# Leaf Disease Detection using Deep Learning
## Capstone Report

**Author:** Mekha S R

---

# 1. Introduction

Plant diseases significantly reduce crop productivity and quality. Early detection enables farmers to take timely preventive measures, reducing economic losses and improving agricultural sustainability.

This project develops an automated leaf disease detection system using Deep Learning and Computer Vision. A Convolutional Neural Network (CNN) and Transfer Learning with ResNet18 were implemented to classify healthy and diseased leaves from images.

The final system includes:

- Image preprocessing
- Deep learning inference
- REST API deployment using FastAPI
- Interactive documentation using Swagger UI

---

# 2. Agritech Problem

Manual disease identification requires agricultural expertise and is often unavailable to small-scale farmers.

Challenges include:

- Similar symptoms across diseases
- Lighting variations
- Background clutter
- Different camera qualities
- Large number of crop species

The objective is to build a lightweight AI model capable of automatically identifying leaf diseases from field images.

---

# 3. Dataset

## Classes

The dataset contains images from multiple crop categories.

Example classes:

- Chili Healthy
- Chili Leaf Curl
- Chilli Leaf Spot
- Tomato Healthy
- Tomato Bacterial Spot
- Tomato Late Blight

Images were divided into:

- Training set
- Validation set
- Test set

Each image was resized to:

```
224 × 224
```

to match ResNet18 input dimensions.

---

# 4. Data Preprocessing

The following preprocessing steps were applied:

- Resize (224×224)
- Convert to Tensor
- Normalize using ImageNet statistics

Training augmentations included:

- Random Horizontal Flip
- Random Rotation (15°)
- Color Jitter

These augmentations improve model robustness against real-world variations.

---

# 5. Model Architecture

## Baseline CNN

Initially, a custom CNN was implemented to understand the complete deep learning workflow.

Architecture included:

- Convolution Layers
- ReLU
- MaxPooling
- Fully Connected Layer
- Softmax Classification

---

## Transfer Learning

To improve performance, pretrained ResNet18 was used.

Reasons:

- Faster convergence
- Better feature extraction
- Higher accuracy
- Reduced training time

The final classification layer was replaced according to the number of disease classes.

---

# 6. Training Configuration

Loss Function

```
CrossEntropyLoss
```

Optimizer

```
Adam
```

Learning Rate

```
0.001
```

Batch Size

```
32
```

Epochs

```
10
```

Transfer learning significantly reduced overfitting compared to training from scratch.

---

# 7. Model Evaluation

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics provide a balanced evaluation across all disease classes.

---

# 8. Error Analysis

The confusion matrix revealed that most errors occurred between visually similar disease categories.

Observed issues:

- Healthy leaves misclassified under poor lighting
- Similar disease symptoms confused
- Background vegetation affected predictions
- Limited samples for certain classes

Strong Performing Classes:

- Tomato Healthy
- Tomato Late Blight

Weak Performing Classes:

- Chili Healthy
- Chili Leaf Curl
- Chilli Leaf Spot

Reasons:

- Small dataset
- High visual similarity
- Uneven class distribution

---

# 9. Model Optimization

Several improvements were explored:

- Data Augmentation
- Transfer Learning
- Better preprocessing
- TorchScript model export

Future improvements:

- Larger dataset
- More crop species
- Mobile deployment
- Edge AI optimization
- Attention-based CNNs

---

# 10. TorchScript Export

The trained ResNet18 model was converted into TorchScript.

Benefits:

- Faster inference
- Portable deployment
- Language-independent execution
- Production-ready model

Output:

```
resnet18_scripted.pt
```

---

# 11. FastAPI Deployment

A REST API was developed using FastAPI.

Available endpoints:

## GET /

Returns API status.

## POST /predict

Accepts:

- Image file

Returns:

```json
{
    "prediction":"Tomato___Late_blight",
    "confidence":0.97
}
```

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

---

# 12. Deployment Workflow

```
User
   │
   ▼
Upload Leaf Image
   │
   ▼
FastAPI Server
   │
   ▼
Image Preprocessing
   │
   ▼
TorchScript Model
   │
   ▼
Prediction
   │
   ▼
JSON Response
```

---

# 13. Ethics and Privacy

Although this project focuses on crop disease detection, responsible AI practices remain important.

## Privacy

- Images should not contain identifiable human faces.
- Farmer information should never be stored without consent.
- Uploaded images should be processed securely.

## Bias

The model performance depends on dataset diversity.

Potential bias may arise due to:

- Limited crop varieties
- Geographic imbalance
- Seasonal differences

Future datasets should include wider environmental conditions.

## Responsible Use

The model should assist agricultural experts rather than replace professional diagnosis.

Predictions should be treated as decision support instead of final medical advice for crops.

---

# 14. Limitations

Current limitations include:

- Limited dataset size
- Restricted disease categories
- Controlled image conditions
- CPU inference only
- No mobile deployment

---

# 15. Future Work

Potential improvements include:

- Mobile application
- Real-time camera inference
- Cloud deployment
- Edge AI optimization
- Explainable AI using Grad-CAM
- Multi-crop disease support
- Automatic treatment recommendation

---

# 16. Conclusion

This project demonstrates an end-to-end Computer Vision pipeline for automated leaf disease detection.

Starting from dataset preparation, the workflow progressed through CNN implementation, transfer learning, data augmentation, evaluation, TorchScript conversion, and FastAPI deployment.

The resulting system provides an efficient and scalable solution that can assist farmers and agricultural experts in early disease detection, while serving as a strong foundation for future smart agriculture applications.

---

# Project Workflow

```
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Data Augmentation
   │
   ▼
ResNet18 Training
   │
   ▼
Evaluation
   │
   ▼
TorchScript Export
   │
   ▼
FastAPI Deployment
   │
   ▼
Leaf Disease Prediction
```

---