import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-e", "--pretrain_epochs", type=int)
    parser.add_argument("-t", "--train_epochs", type=int)
    args = parser.parse_args()

    batch_size = args.batch_size
    pretrain = args.pretrain
    pretrain_epochs = args.pretrain_epochs
    train_epochs = args.train_epochs

    return batch_size, pretrain, pretrain_epochs, train_epochs


def get_training_name(batch_size, pretrain_epochs, train_epochs, test, extra=None):
    pretrain_str = f"Pretrain{pretrain_epochs}" 
    train_str = f"Train{train_epochs}"
    batch_size_str = f"Batch{batch_size}"
    training_name = f"{pretrain_str}_{train_str}_{batch_size_str}"

    if extra:
        training_name = extra + training_name
    if test:
        training_name = "TEST_" + training_name

    log_dir = "./logs/" + training_name

    print()
    print(training_name)
    print()
    print(f"batch size: {batch_size}, type(batch_size): {type(batch_size)})")
    print(f"pretrain_epochs: {pretrain_epochs}, type(pretrain_epochs): {type(pretrain_epochs)})")
    print(f"train_epochs: {train_epochs}, type(train_epochs): {type(train_epochs)})")
    print()

    return training_name, log_dir


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
        "./data", download=True, transform=torchvision.transforms.ToTensor(), split="train",
    )
    val_data = torchvision.datasets.STL10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), split="test"
    )

    loader, val_loader = get_dataloaders(data, val_data, batch_size, test)

    return loader, val_loader
