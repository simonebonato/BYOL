import torch
from BYOL import BYOL
from model import Model
from utils import get_loaders_CIFAR10, get_loaders_STL10

torch.manual_seed(42)

test = 2000
device = "cuda"

batch_size = 100
pretrain_epochs = 1001
train_epochs = 100
lin_evaluation_frequency = 50
log_dir = "./logs/tests/CIFAR10/meeting_test-2000_samples"

loader, val_loader = get_loaders_CIFAR10(batch_size, test=test)
# loader, val_loader = get_loaders_STL10(batch_size, test=test)

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
    hidden_features=4096,
    device=device,
    lin_evaluation_frequency=lin_evaluation_frequency,
)
byol.pretrain()
