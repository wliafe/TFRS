from torch.utils import data
from torchvision import datasets
from pathlib import Path


def load_image_classification_data(train_path, test_path, train_transform, test_transform, batch_size=100):
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    test_data = datasets.ImageFolder(test_path, transform=test_transform)
    train_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_iter = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_iter, test_iter, train_data.classes


def load_image_labels_data(train_path):
    folder_path = Path(train_path)
    return [label.name for label in folder_path.iterdir()]
