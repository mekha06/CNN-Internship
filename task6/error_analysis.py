import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


val_dir = "../data/val"
model_path = "../task5/models/task5_resnet18_augmented.pth"

os.makedirs("reports", exist_ok=True)
os.makedirs("reports/errors", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = val_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

true_labels = []
pred_labels = []
wrong_images = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

        for i in range(len(labels)):
            if labels[i] != predictions[i]:
                wrong_images.append({
                    "image": images[i].cpu(),
                    "true": labels[i].item(),
                    "pred": predictions[i].item()
                })

report = classification_report(
    true_labels,
    pred_labels,
    target_names=class_names
)

print(report)

with open("reports/classification_report.txt", "w") as file:
    file.write(report)

cm = confusion_matrix(true_labels, pred_labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for index, item in enumerate(wrong_images[:5]):
    image = item["image"] * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()

    true_class = class_names[item["true"]]
    pred_class = class_names[item["pred"]]

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"True: {true_class}\nPredicted: {pred_class}")
    plt.tight_layout()
    plt.savefig(f"reports/errors/error_{index + 1}.png")
    plt.close()

analysis = f"""# Task 6 Error Analysis

## Objective

The objective of this task was to evaluate the trained leaf disease classification model using confusion matrix, classification report, and misclassified image analysis.

## Methodology

- Loaded the validation dataset using ImageFolder.
- Loaded the trained ResNet18 model.
- Predicted labels for validation images.
- Compared actual labels with predicted labels.
- Generated a confusion matrix.
- Generated a classification report.
- Saved misclassified images inside reports/errors.

## Classes Evaluated

{class_names}

## Observations

- The confusion matrix shows correct and incorrect predictions for each class.
- The diagonal values represent correct predictions.
- The non-diagonal values represent misclassifications.
- The classification report shows precision, recall, F1-score, and support for each class.

## Error Analysis

- Some leaves may be misclassified because healthy and diseased leaves can look visually similar.
- Disease symptoms may be small, unclear, or partially visible.
- Lighting, shadows, background noise, and leaf angle can affect predictions.
- Similar disease patterns can confuse the model.

## Possible Improvements

- Add more images for weak classes.
- Use stronger data augmentation.
- Train for more epochs if needed.
- Fine-tune more ResNet18 layers.
- Use balanced class distribution.

## Conclusion

The model was evaluated using numerical metrics and visual error analysis. The confusion matrix and classification report helped understand class-wise performance, while misclassified samples helped identify practical reasons for errors.
"""

with open("reports/error_analysis.md", "w") as file:
    file.write(analysis)

print("Confusion matrix saved at reports/confusion_matrix.png")
print("Classification report saved at reports/classification_report.txt")
print("Misclassified samples saved at reports/errors")
print("Error analysis saved at reports/error_analysis.md")
print("Task 6 completed successfully.")