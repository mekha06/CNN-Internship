# Task 6: Confusion Matrix & Error Analysis

## Overview
Task 6 focuses on evaluating the trained CNN (ResNet18) model using detailed performance metrics. The goal is to understand not just accuracy, but also **where and why the model fails**.

This task provides deeper insight using:
- Confusion Matrix
- Classification Report
- Misclassified Image Analysis

---

## Objective
To evaluate the trained leaf disease classification model and analyze:
- Class-wise performance
- Model confusion between similar diseases
- Weak areas of the model using error analysis

---

## Methodology

### 1. Model Loading
- Loaded the trained ResNet18 model from Task 5
- Set model to evaluation mode (`model.eval()`)

### 2. Validation Dataset
- Used `ImageFolder` for loading validation data
- Applied same preprocessing as training (Resize + ToTensor)

### 3. Prediction Process
- Passed validation images through the model
- Collected:
  - True labels
  - Predicted labels

### 4. Evaluation Metrics
Generated the following:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Accuracy score

### 5. Error Analysis
- Extracted misclassified images
- Saved top incorrect predictions
- Visual inspection of failure cases

---

## Dataset Classes

```python id="t6classes"
['Chili___healthy',
 'Chili___leaf_curl',
 'Chilli__leaf_spot',
 'Tomato___Bacterial_spot',
 'Tomato___Late_blight',
 'Tomato___healthy'] '''

## Strong Performing Classes

- Tomato___Late_blight  
- Tomato___healthy  

### Why these classes perform better:
- Clear visual patterns  
- Better feature representation  
- More consistent training samples  

---

## Weak Performing Classes

- Chili___healthy  
- Chili___leaf_curl  
- Chilli__leaf_spot  

### Issues observed:
- Low recall  
- Frequent misclassification into Tomato classes  
- Poor feature separation  
# need to improve dataset since chilli class contains lees no images so need to add more dta and implement augmentation too