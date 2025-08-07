import mltools
from torchvision import datasets


def tfrs(transform, path="../data", batch_size=8, num_workers=8, pin_memory=True, drop_last=True):
    data = datasets.ImageFolder(f"{path}/TFRS", transform=transform)
    train_data, val_data, test_data = mltools.split_data(data, [0.7, 0.15, 0.15])
    return mltools.iter_data([train_data, val_data, test_data], batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
