from pathlib import Path

import torch
from BYOL import BYOL
from model import Model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import argparser, get_loaders_CIFAR10, get_training_name

torch.manual_seed(42)

test = False
device = "cuda"

# pretrain = True
# only_cnn = False

# batch_size = 300
# pretrain_epochs = 20
# train_epochs = 20

batch_size, pretrain, only_cnn, pretrain_epochs, train_epochs = argparser()
training_name = get_training_name(batch_size, pretrain, only_cnn, pretrain_epochs, train_epochs)
loader, val_loader = get_loaders_CIFAR10(batch_size, test=test)

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

    models_folder = Path("./models")
    if not models_folder.exists():
        models_folder.mkdir()
    torch.save(model.state_dict(), f"./models/{training_name}.pth")


# ############ TRAINING ############
print("Training")

logger = SummaryWriter(log_dir="./logs/" + training_name, comment="CIFAR10")

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-5)
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

