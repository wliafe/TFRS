from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path


def load_image_classification_data(path, transform, batch_size):
    data = datasets.ImageFolder(path, transform=transform)
    iters = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return iters


def load_image_labels_data(train_path):
    folder_path = Path(train_path)
    return [label.name for label in folder_path.iterdir()]
