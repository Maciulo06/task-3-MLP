from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(normalize=True):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    return transforms.Compose(transform_list)


def load_datasets(data_dir='data', val_ratio=0.1, normalize=True):
    base_transform = get_transforms(normalize=normalize)
    analysis_transform = transforms.ToTensor()

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=base_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=base_transform)
    analysis_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=analysis_transform)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)
    return train_set, val_set, test_set, analysis_train


def create_loaders(train_set, val_set, test_set, batch_size=128):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def dataset_statistics(dataset):
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    return {
        'shape': tuple(images.shape),
        'min_pixel': float(images.min().item()),
        'max_pixel': float(images.max().item()),
        'mean': images.mean(dim=(0, 2, 3)).tolist(),
        'std': images.std(dim=(0, 2, 3)).tolist(),
        'class_counts': dict(class_counts)
    }


def get_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']