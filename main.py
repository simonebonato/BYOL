# %%
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from BYOL import BYOL
from model import Model

%load_ext autoreload
%autoreload 2

torch.manual_seed(42)

training_name = "pretrain-cnn-60"
pretrain = True
only_cnn = True

batch_size = 200
pretrain_epochs = 5
train_epochs = 40
device = "mps"

data = torchvision.datasets.CIFAR10("./data", download=True, transform=torchvision.transforms.ToTensor(), train=True)
val_data = torchvision.datasets.CIFAR10(
    "./data", download=True, transform=torchvision.transforms.ToTensor(), train=False
)

loader = DataLoader(data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = Model(3, 10).to(device)
model.train()


# ############ PRETRAINING ############
if pretrain:
    print("Pretraining")
    train_model = model.cnn if only_cnn else model
    byol = BYOL(
        train_model, loader, pretrain_epochs, input_dims=3, img_dims=(32, 32), hidden_features=2048, device=device
    )
    byol.pretrain()

    models_folder = Path("./models").mkdir(exist_ok=True)
    only_cnn_str = "_only_cnn" if only_cnn else ""
    pretrain_epochs_str = f"_pretrain_epochs_{pretrain_epochs}" if pretrain_epochs > 0 else ""
    torch.save(model.state_dict(), f"./models/pretrained{only_cnn_str}{pretrain_epochs_str}_{training_name}.pth")


# %%
# ############ TRAINING ############
print("Training")

logger = SummaryWriter(log_dir="./logs/" + training_name, comment="CIFAR10")

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
for epoch in range(train_epochs):
    # # train loop
    loop = tqdm(loader)
    losses = []
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()

        out = model(x)
        l = loss(out, y)
        l.backward()
        opt.step()

        losses.append(l.item())
        loop.set_description(f"Train: Epoch {epoch} - Loss: {l.item():.2f}")

    logger.add_scalar("Loss/train", sum(losses) / len(losses), epoch)

    # val loop
    loop = tqdm(val_loader)
    losses = []
    accuracies = []
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        l = loss(out, y)

        acc = (out.argmax(dim=1) == y).float().sum()
        accuracies.append(acc.item())

        losses.append(l.item())
        loop.set_description(f"Val: Epoch {epoch} - Loss: {l.item():.2f}")

    logger.add_scalar("Loss/val", sum(losses) / len(losses), epoch)
    logger.add_scalar("Accuracy/val", sum(accuracies) / len(loader.dataset), epoch)
