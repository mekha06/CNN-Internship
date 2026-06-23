# Task 8: FastAPI Inference Endpoint

## Objective

The objective of this task is to deploy the trained **ResNet18 leaf disease classification model** as a REST API using **FastAPI**. The API allows users to upload a leaf image and receive the predicted disease class, confidence score, and inference time.

---

## Features

* REST API built with FastAPI
* Upload leaf images using `multipart/form-data`
* Predict disease class using the trained ResNet18 model
* Returns confidence score and inference time
* Health check endpoint
* Interactive Swagger UI for testing
* Input validation for supported image formats

---

## Project Structure

```text
leaf-disease-cv/
│
├── api/
│   └── main.py
│
├── models/
│   └── resnet18_leaf_disease.pth
│
├── src/
├── requirements.txt
└── README.md
```

---

## Installation

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision
```

---

## Running the API

Navigate to the project root directory and start the server.

```bash
python -m uvicorn api.main:app --reload
```

The API will start at

```text
http://127.0.0.1:8000
```

---

## Interactive API Documentation

Swagger UI

```text
http://127.0.0.1:8000/docs
```

ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# API Endpoints

## 1. Root Endpoint

**GET /**

Returns a welcome message.

### Example Response

```json
{
    "message": "Leaf Disease Prediction API"
}
```

---

## 2. Health Check

**GET /health**

Checks whether the API is running correctly.

### Example Response

```json
{
    "status": "healthy"
}
```

---

## 3. Disease Prediction

**POST /predict**

Uploads a leaf image and predicts the disease class.

### Request

| Parameter | Type                    | Description               |
| --------- | ----------------------- | ------------------------- |
| file      | Image (.jpg/.jpeg/.png) | Leaf image for prediction |

### Example Response

```json
{
    "predicted_class": "Tomato___Late_blight",
    "confidence": 98.73,
    "inference_time_ms": 21.45
}
```

---

# Testing Using cURL

The prediction endpoint can be tested directly from the terminal.

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
-H "accept: application/json" ^
-H "Content-Type: multipart/form-data" ^
-F "file=@sample_leaf.jpg"
```

### Linux/macOS

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@sample_leaf.jpg"
```

---

# Supported Disease Classes

* Chili Healthy
* Chili Leaf Curl
* Chili Leaf Spot
* Tomato Bacterial Spot
* Tomato Late Blight
* Tomato Healthy

---

# Model Information

| Property          | Value     |
| ----------------- | --------- |
| Model             | ResNet18  |
| Framework         | PyTorch   |
| Input Size        | 224 × 224 |
| Transfer Learning | Yes       |
| Number of Classes | 6         |

---

# Image Preprocessing

Before inference, every uploaded image undergoes the following preprocessing steps:

* Convert image to RGB
* Resize to **224 × 224**
* Convert image to tensor
* Normalize using ImageNet mean and standard deviation

---

# Error Handling

The API validates uploaded files and returns appropriate HTTP responses.

| Error                 | HTTP Status |
| --------------------- | ----------- |
| Invalid image         | 400         |
| Unsupported file type | 400         |
| Missing uploaded file | 422         |

---

# Example Workflow

1. Start the FastAPI server.
2. Open `http://127.0.0.1:8000/docs`.
3. Select the **POST /predict** endpoint.
4. Upload a leaf image (`.jpg`, `.jpeg`, or `.png`).
5. Execute the request.
6. Receive the predicted disease class, confidence score, and inference time.

---

# Technologies Used

* Python
* FastAPI
* Uvicorn
* PyTorch
* Torchvision
* Pillow
* python-multipart

---

# Outcome

The trained ResNet18 model was successfully deployed as a FastAPI inference service. The API accepts image uploads, performs preprocessing, runs model inference, and returns the predicted disease class with confidence and inference time through REST endpoints. This deployment demonstrates how a deep learning model can be exposed as a production-ready prediction service.
