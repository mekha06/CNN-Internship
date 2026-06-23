from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image, UnidentifiedImageError
from io import BytesIO

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

import time

app = FastAPI(
    title="Leaf Disease Classification API",
    description="Predicts the disease class of a leaf image using a trained ResNet18 model.",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    "Chili___healthy",
    "Chili___leaf_curl",
    "Chilli__leaf_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(
    torch.load(
        "../task5/models/task5_resnet18_augmented.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def home():
    return {
        "message": "Leaf Disease Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    start = time.time()

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    end = time.time()

    return JSONResponse(
        content={
            "predicted_class": classes[prediction.item()],
            "confidence": round(confidence.item() * 100, 2),
            "inference_time_ms": round((end - start) * 1000, 2)
        }
    )