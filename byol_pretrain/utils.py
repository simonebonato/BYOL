import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_tb_logger(log_dir, extra_str=None):
    if extra_str:
        log_dir = log_dir + extra_str
    logger = SummaryWriter(log_dir=log_dir, comment="CIFAR10")
    return logger


def get_dataloaders(data, val_data, batch_size, test):
    if test is not None:
        data = torch.utils.data.Subset(data, range(min(test, len(data))))
        val_data = torch.utils.data.Subset(val_data, range(min(test, len(data))))

    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return loader, val_loader


def get_loaders_CIFAR10(batch_size, test=None):
    data = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), train=True
    )
    val_data = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), train=False
    )

    loader, val_loader = get_dataloaders(data, val_data, batch_size, test)

    return loader, val_loader


def get_loaders_STL10(batch_size, test=None):
    data = torchvision.datasets.STL10(
        "./data",
        download=True,
        transform=torchvision.transforms.ToTensor(),
        split="train",
    )
    val_data = torchvision.datasets.STL10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), split="test"
    )

    loader, val_loader = get_dataloaders(data, val_data, batch_size, test)

    return loader, val_loader
