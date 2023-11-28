from pathlib import Path

import torch
from augmentations import ByolAugmentations
from BYOL import BYOL
from model import Model
from tqdm import tqdm
from utils import argparser, get_loaders_CIFAR10, get_training_name

torch.manual_seed(42)

test = False
use_argparser = False
device = "cuda"

batch_size = 300
pretrain_epochs = 1000
train_epochs = 100
lin_evaluation_frequency=80

# get the logdir name
training_name, log_dir = get_training_name(batch_size, pretrain_epochs, train_epochs, test)
loader, val_loader = get_loaders_CIFAR10(batch_size, test=test)

model = Model(3).to(device)
model.train()

# ############ PRETRAINING ############

print("Pretraining")
byol = BYOL(
    model,
    loader,
    val_loader,
    pretrain_epochs,
    log_dir,
    input_dims=3,
    img_dims=(32, 32),
    hidden_features=2048,
    device=device,
    lin_evaluation_frequency=lin_evaluation_frequency,
)
byol.pretrain()

models_folder = Path("./models")
if not models_folder.exists():
    models_folder.mkdir()
torch.save(model.state_dict(), f"./models/{training_name}.pth")


# ############ TRAINING ############
# actually not strictly necessary, but can be useful later
# print("Training")

# loss = torch.nn.CrossEntropyLoss()
# opt = torch.optim.Adam(model.parameters(), lr=3e-5)
# augs = ByolAugmentations((32, 32)).get_augmentations()[0]
# for epoch in range(train_epochs):
#     # # train loop
#     loop = tqdm(loader)
#     losses = []
#     accuracies = []
#     for x, y in loop:
#         x = (augs(x)).to(device)
#         y = y.to(device)

#         opt.zero_grad()

#         out = model(x)
#         l = loss(out, y)
#         l.backward()
#         opt.step()

#         acc = (out.argmax(dim=1) == y).float().sum()
#         accuracies.append(acc.item())

#         losses.append(l.item())
#         loop.set_description(f"Train: Epoch {epoch} - Loss: {l.item():.2f} - Acc: {acc.item():.2f}")

#     logger.add_scalar("Loss/train", sum(losses) / len(losses), epoch)
#     logger.add_scalar("Accuracy/train", sum(accuracies) / len(loader.dataset), epoch)

#     # val loop
#     loop = tqdm(val_loader)
#     losses = []
#     accuracies = []
#     for x, y in loop:
#         x = x.to(device)
#         y = y.to(device)

#         out = model(x)
#         l = loss(out, y)

#         acc = (out.argmax(dim=1) == y).float().sum()
#         accuracies.append(acc.item())

#         losses.append(l.item())
#         loop.set_description(f"Val: Epoch {epoch} - Loss: {l.item():.2f} - Acc: {acc.item():.2f}")

#     logger.add_scalar("Loss/val", sum(losses) / len(losses), epoch)
#     logger.add_scalar("Accuracy/val", sum(accuracies) / len(val_loader.dataset), epoch)
