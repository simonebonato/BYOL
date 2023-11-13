# %%
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm


class BYOL(nn.Module):
    def __init__(
        self,
        model,
        dataloader,
        num_epochs,
        tau_base=0.99,
        input_dims=1,
        img_dims=(768, 1024),
        device="mps",
        hidden_features=4096,
        output_features=256,
    ):
        super(BYOL, self).__init__()

        self.student = model.to(device)
        self.teacher = copy.deepcopy(model).to(device)

        # freeze the teacher weights, only updates with EMA
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=3e-4)

        self.dataloader = dataloader

        self.num_epochs = num_epochs
        self.total_steps = len(dataloader) * num_epochs

        self.tau = tau_base
        self.tau_base = tau_base

        self.input_dims = input_dims
        self.img_dims = img_dims
        self.device = device
        self.hidden_features = hidden_features
        self.output_features = output_features

        self.projection_head = self.get_projection_head().to(device)

        self.aug1 = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomRotation((-15, 15)),
                T.GaussianBlur(3, sigma=(0.1, 2.0)),
            ]
        )
        self.aug2 = T.Compose(
            [
                T.RandomAffine(translate=(0.1, 0.2), degrees=(5, 15), shear=(5, 15)),
            ]
        )

    def get_projection_head(self):
        mock_input = torch.randn(1, self.input_dims, *self.img_dims).to(self.device)
        mock_output = self.student(mock_input)

        projection_head = nn.Sequential(
            nn.Linear(mock_output.shape[-1], self.hidden_features),
            nn.BatchNorm2d(mock_output.shape[1]) if len(mock_output.shape) > 2 else nn.BatchNorm1d(mock_output.shape[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.output_features),
        )
        return projection_head

    def update_teacher(self):
        for name, teacher_weight in self.teacher.named_parameters():
            teacher_weight.data = self.tau * teacher_weight.data + (1 - self.tau) * self.student.state_dict()[name].data

    def update_tau(self, step):
        self.tau = 1 - (1 - self.tau_base) * ((np.cos((np.pi * step) / self.total_steps) + 1) / 2)

    def augment(self, x):
        view1 = self.aug1(x)
        view2 = self.aug1(x)
        return view1, view2

    def loss_fn(self, student_out, teacher_out):
        student_out = F.normalize(student_out, dim=-1, p=2)
        teacher_out = F.normalize(teacher_out, dim=-1, p=2)

        loss = 2 - 2 * (student_out * teacher_out).sum(dim=-1)
        return loss.mean()

    def forward_pass(self, x):
        view1, view2 = self.augment(x)

        student_out_1 = self.projection_head(self.student(view1))
        student_out_2 = self.projection_head(self.student(view2))

        teacher_out1 = self.projection_head(self.teacher(view1))
        teacher_out2 = self.projection_head(self.teacher(view2))

        loss = self.loss_fn(student_out_1, teacher_out1) + self.loss_fn(student_out_2, teacher_out2)

        return loss

    def backward_pass(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pretrain(self):
        step_counter = 0
        loop = tqdm(range(self.num_epochs))
        for epoch in loop:
            for idx, (x, _) in enumerate(self.dataloader):
                loop.set_description(
                    f"Epoch {epoch + 1} / {self.num_epochs} | Step {idx} / {len(self.dataloader)} | Tau: {self.tau:.4f}"
                )
                x = x.to(self.device)
                loss = self.forward_pass(x)
                self.backward_pass(loss)

                self.update_teacher()
                self.update_tau(step_counter)
                step_counter += 1


# %%

if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    from model import Model

    model = Model(3, 10)
    original_weights = copy.deepcopy(model.state_dict())

    data = torchvision.datasets.CIFAR10(
        "./data", download=True, transform=torchvision.transforms.ToTensor(), train=True
    )

    loader = DataLoader(data, batch_size=300)
    byol = BYOL(model, loader, 2, input_dims=3, img_dims=(32, 32))
    byol.pretrain()
# %%
