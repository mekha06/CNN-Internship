import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Folder paths
train_dir = "../data/train"
val_dir = "../data/val"
output_dir = "outputs"
model_dir = "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Augmentation for training data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# No augmentation for validation data
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
class_names = train_data.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# Function to get one image path from training folder
def get_sample_image(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(root, file)

# Visualize augmentation stages
sample_path = get_sample_image(train_dir)
image = Image.open(sample_path).convert("RGB")
resize = transforms.Resize((224, 224))
flip = transforms.RandomHorizontalFlip(p=1)
rotate = transforms.RandomRotation(15)
color = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
original = resize(image)
flipped = flip(original)
rotated = rotate(original)
color_changed = color(original)

images = [original, flipped, rotated, color_changed]
titles = ["Original", "Flip", "Rotation", "Color jitter"]

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/task5_augmentation_grid.png")
plt.show()

print("Augmentation grid saved at outputs/task5_augmentation_grid.png")

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze old ResNet18 layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for our classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Lists to store graph values
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.4f}")
    print(f"Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "models/task5_resnet18_augmented.pth")
print("Model saved at models/task5_resnet18_augmented.pth")

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.title("Loss Curve With Augmentation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/task5_loss_curve.png")
plt.show()

# Plot accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train accuracy")
plt.plot(val_accs, label="Validation accuracy")
plt.title("Accuracy Curve With Augmentation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("outputs/task5_accuracy_curve.png")
plt.show()
print("Training curves saved in outputs folder")