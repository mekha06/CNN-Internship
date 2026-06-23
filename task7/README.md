# Task 7 – Model Export and Inference

## Objective

The objective of this task is to export the trained ResNet18 model for inference and create a command-line prediction script that can classify new leaf images without retraining the model.

---

## Methodology

### Step 1: Load the Trained Model

* Loaded the trained ResNet18 model architecture.
* Modified the final fully connected layer to output 6 classes.
* Loaded the trained weights from `task5/models/task5_resnet18_augmented.pth`.

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("task5/models/task5_resnet18_augmented.pth"))
```

---

### Step 2: Export the Model

* Set the model to evaluation mode.
* Created a dummy input tensor of size **1 × 3 × 224 × 224**.
* Converted the model to TorchScript using `torch.jit.trace()`.
* Saved the exported model as `task7/resnet18_scripted.pt`.

---

### Step 3: Build the Prediction Script

Created `predict.py` to:

* Load the exported TorchScript model.
* Read an input image.
* Resize the image to **224 × 224**.
* Convert the image into a tensor.
* Perform inference.
* Apply the Softmax function to obtain prediction probabilities.
* Display the predicted class and confidence score.

---

## Model Information

**Model:** ResNet18 (Transfer Learning)

**Framework:** PyTorch

**Export Format:** TorchScript (`.pt`)

---

## Input Requirements

| Parameter         | Value       |
| ----------------- | ----------- |
| Image Size        | 224 × 224   |
| Color Mode        | RGB         |
| Tensor Conversion | Yes         |
| Normalization     | Not Applied |

> **Note:** During Task 5 training, normalization was not used. Therefore, the same preprocessing was followed during inference to ensure consistent predictions.

---

## Class Mapping

| Index | Class                   |
| ----: | ----------------------- |
|     0 | Chili___healthy         |
|     1 | Chili___leaf_curl       |
|     2 | Chilli__leaf_spot       |
|     3 | Tomato___Bacterial_spot |
|     4 | Tomato___Late_blight    |
|     5 | Tomato___healthy        |

---

## Running the Prediction Script

```bash
python task7/predict.py path/to/image.jpg
```

Example:

```bash
python task7/predict.py "data/train/Tomato___healthy/example.jpg"
```

---

## Sample Output

```
Prediction
---------------------
Class      : Tomato___healthy
Confidence : 98.73%
```

---

## Testing

The prediction script was tested on multiple leaf images from different classes.

The model successfully:

* Loaded the exported TorchScript model.
* Processed input images correctly.
* Predicted the corresponding disease class.
* Displayed the prediction confidence.

---

# 
The model is biased towards chilli dataset and performs badly on tomato , need to make improvements to make the model strong and accurate. Gonna work on that!!
need to make improvements on data , augmentation , model training etc

## Conclusion

The trained ResNet18 model was successfully exported into TorchScript format for deployment. A command-line inference script was implemented to classify unseen leaf images. The preprocessing pipeline used during inference was kept consistent with the training pipeline, resulting in correct predictions on test images.
