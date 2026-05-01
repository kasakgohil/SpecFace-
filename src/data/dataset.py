import os
from torchvision import datasets, transforms

def get_torchvision_datasets(data_dir, train_transform, val_transform=None):
    if val_transform is None:
        val_transform = train_transform

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset
