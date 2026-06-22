import json
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ----------------------------
# Class Names
# ----------------------------
class_names = [
    "Chili___healthy",
    "Chili___leaf_curl",
    "Chilli__leaf_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model = torch.jit.load(
    "task7/resnet18_scripted.pt",
    map_location=device
)

model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Read Image
# ----------------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

print("\nPrediction")
print("---------------------")
print(f"Class      : {class_names[predicted.item()]}")
print(f"Confidence : {confidence.item()*100:.2f}%")