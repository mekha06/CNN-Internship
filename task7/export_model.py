import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 6

device = torch.device("cpu")

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(
    torch.load(
        "task5/models/task5_resnet18_augmented.pth",
        map_location=device
    )
)

model.eval()

example_input = torch.randn(1, 3, 224, 224)

scripted_model = torch.jit.trace(model, example_input)

scripted_model.save("task7/resnet18_scripted.pt")

print("TorchScript model exported successfully!")