import os
import matplotlib.pyplot as plt

from src.dataset import create_dataloaders, get_class_counts


train_dir = "data/train"
val_dir = "data/val"

train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    batch_size=16,
    num_workers=0
)

print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)

print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))

print("\nClass imbalance summary:")
class_counts = get_class_counts(train_dataset)

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

images, labels = next(iter(train_loader))

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)

os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(4, 4, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    image = images[i].permute(1, 2, 0)
    label = labels[i].item()
    class_name = train_dataset.classes[label]

    ax.imshow(image)
    ax.set_title(class_name, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_batch.png")
plt.show()