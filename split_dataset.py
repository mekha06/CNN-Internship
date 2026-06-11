import os
import random
import shutil


# Original downloaded tomato dataset folder
# This folder should contain Tomato___Late_blight and Tomato___healthy folders
source_root = r"D:\tomato-disease"

# Project data folder
destination_root = r"D:\cnn-internship\leaf-disease-cv\data"

# Tomato classes you want to use
classes = [
    "Tomato___Late_blight",
    "Tomato___healthy" ,
    "Tomato___Bacterial_spot"
]

# Number of images per class
train_count = 200
val_count = 50

# To get the same random split every time
random.seed(42)


for class_name in classes:
    source_dir = os.path.join(source_root, class_name)

    train_dir = os.path.join(destination_root, "train", class_name)
    val_dir = os.path.join(destination_root, "val", class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [
        file for file in os.listdir(source_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]

    for image in train_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(train_dir, image)
        shutil.copy2(source_path, destination_path)

    for image in val_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(val_dir, image)
        shutil.copy2(source_path, destination_path)

    print(f"{class_name} completed")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print("-" * 40)


print("Tomato classes split successfully.")