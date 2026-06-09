import os
from collections import Counter
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            folder_name for folder_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder_name))
        ])

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)

            for image_name in os.listdir(class_folder):
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_folder, image_name)
                    label = self.class_to_idx[class_name]

                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(train_dir, val_dir, batch_size=16, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = LeafDiseaseDataset(train_dir, transform=transform)
    val_dataset = LeafDiseaseDataset(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataset, val_dataset, train_loader, val_loader


def get_class_counts(dataset):
    counts = Counter(dataset.labels)

    class_counts = {}
    for class_index, count in counts.items():
        class_name = dataset.classes[class_index]
        class_counts[class_name] = count

    return class_counts