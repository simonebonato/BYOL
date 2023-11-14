import argparse

import torch
import torchvision
from torch.utils.data import DataLoader


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-pre", "--pretrain", action="store_true")
    parser.add_argument("-cnn", "--only_cnn", action="store_true")
    parser.add_argument("-e", "--pretrain_epochs", type=int)
    parser.add_argument("-t", "--train_epochs", type=int)
    args = parser.parse_args()

    batch_size = args.batch_size
    pretrain = args.pretrain
    only_cnn = args.only_cnn
    pretrain_epochs = args.pretrain_epochs
    train_epochs = args.train_epochs

    return batch_size, pretrain, only_cnn, pretrain_epochs, train_epochs


def get_training_name(batch_size, pretrain, only_cnn, pretrain_epochs, train_epochs):
    pretrain_str = f"Pretrain{pretrain_epochs}" if pretrain else "NoPretrain"
    only_cnn_str = "OnlyCNN" if (only_cnn and pretrain) else ""
    train_str = f"Train{train_epochs}"
    batch_size_str = f"Batch{batch_size}"
    training_name = f"{pretrain_str}{only_cnn_str}_{train_str}_{batch_size_str}"

    print(training_name)
    print(f"batch size: {batch_size}, type(batch_size): {type(batch_size)})")
    print(f"pretrain: {pretrain}, type(pretrain): {type(pretrain)})")
    print(f"only_cnn: {only_cnn}, type(only_cnn): {type(only_cnn)})")
    print(f"pretrain_epochs: {pretrain_epochs}, type(pretrain_epochs): {type(pretrain_epochs)})")
    print(f"train_epochs: {train_epochs}, type(train_epochs): {type(train_epochs)})")

    return training_name


def get_loaders_CIFAR10(batch_size, test=False):
    data = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), train=True
    )
    val_data = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), train=False
    )

    # make the dataset smaller for testing
    if test:
        training_name = "TEST_" + training_name
        data = torch.utils.data.Subset(data, range(1000))

    loader = DataLoader(data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return loader, val_loader
