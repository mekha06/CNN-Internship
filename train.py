import os  #os helps Python work with folders and file paths.
import torch #PyTorch is used to create tensors , train the model ,move data to CPU/GPU,save the trained model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#torchvision is useful for image-based deep learning.datasets.ImageFolder will load your image dataset from folders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
train_dir="data/train"
val_dir="data/val"
transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)
num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of classes:", num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes).to(device)
print(model)

images, labels = next(iter(train_loader))

single_image = images[0].unsqueeze(0).to(device)

first_conv_layer = model.features[0]

with torch.no_grad():
    feature_maps = first_conv_layer(single_image)

print("Input image shape:", single_image.shape)
print("Feature maps shape after Conv1:", feature_maps.shape)

feature_maps = feature_maps.squeeze(0).cpu()

plt.figure(figsize=(12, 6))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(feature_maps[i], cmap="gray")
    plt.title(f"Feature Map {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/conv1_feature_maps.png")
plt.close()

print("Saved feature map visualization to outputs/conv1_feature_maps.png")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

model_path = "models/cnn_leaf_model.pth"
torch.save(model.state_dict(), model_path)

print(f"\nModel saved to {model_path}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("outputs/loss_curve.png")
plt.close()
print("Saved loss curve to outputs/loss_curve.png")
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), val_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Curve")
plt.grid(True)
plt.savefig("outputs/accuracy_curve.png")
plt.close()
print("Saved accuracy curve to outputs/accuracy_curve.png")
print("\nTraining completed successfully.")

