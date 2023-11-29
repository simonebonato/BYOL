# %%
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from augmentations import ByolAugmentations
from tqdm import tqdm
from utils import get_tb_logger


class ProjectAndNormalize4D(nn.Module):
    def __init__(self, mock_output, hidden_size, output_size, device):
        super(ProjectAndNormalize4D, self).__init__()
        flattened_shape = mock_output.flatten(start_dim=1).shape

        self.projection_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_shape[1], hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).to(device)

        self.proj_output_shape = self.projection_layer(mock_output).shape

    def forward(self, x):
        return self.projection_layer(x)


class ProjectAndNormalize2D(nn.Module):
    def __init__(self, mock_output, hidden_size, output_size, device):
        super(ProjectAndNormalize2D, self).__init__()

        self.projection_layer = nn.Sequential(
            nn.Linear(mock_output.shape[-1], hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        ).to(device)

        self.proj_output_shape = self.projection_layer(mock_output).shape

    def forward(self, x):
        return self.projection_layer(x)


class LinearEvaluator(nn.Module):
    def __init__(self, mock_input, n_classes, device, lr=3e-4, epochs=30):
        super(LinearEvaluator, self).__init__()
        in_dims = mock_input.flatten(start_dim=1).shape[1]

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_dims, n_classes)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr)
        self.epochs = epochs

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.to(self.device)

    def forward_backward(self, x, y, student_model, backward=True):
        x = x.to(self.device)
        y = y.to(self.device)

        student_out = student_model(x)
        _, loss, preds = self(student_out, y)
        if backward:
            self.backward(loss)
        return loss, preds

    def forward(self, x, y):
        x = self.flatten(x)
        x = self.linear(x)

        loss = self.loss_fn(x, y)
        predictions = torch.argmax(x, dim=1)
        return x, loss, predictions

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BYOL(nn.Module):
    def __init__(
        self,
        pretraining_model,
        dataloader,
        val_dataloader,
        num_epochs,
        logdir,
        lin_evaluation_frequency=10,
        tau_base=0.99,
        input_dims=1,
        n_classes=10,
        img_dims=(768, 1024),
        device="mps",
        hidden_features=2048,
        output_features=256,
    ):
        super(BYOL, self).__init__()

        self.student = pretraining_model.to(device)
        self.teacher = copy.deepcopy(pretraining_model).to(device)

        # freeze the teacher weights, only updates with EMA
        self.change_requires_grad(self.teacher, False)

        # defining objects and variables for the training
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=3e-4)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.lin_evaluation_frequency = lin_evaluation_frequency
        self.total_steps = len(dataloader) * num_epochs
        self.tau, self.tau_base = tau_base, tau_base
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.img_dims = img_dims
        self.device = device
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.logdir = logdir

        self.projection_head = self.get_projection_head().to(device)
        self.aug1, self.aug2 = ByolAugmentations(img_dims).get_augmentations()

    def change_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def get_projection_head(self):
        mock_input = torch.randn(2, self.input_dims, *self.img_dims).to(self.device)
        mock_output = self.student(mock_input)

        self.mock_student_output = mock_output

        if len(mock_output.shape) == 2:
            return ProjectAndNormalize2D(mock_output, self.hidden_features, self.output_features, self.device)
        elif len(mock_output.shape) == 4:
            return ProjectAndNormalize4D(mock_output, self.hidden_features, self.output_features, self.device)

    def update_teacher(self):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            update_weights, old_weight = student_param.data, teacher_param.data
            teacher_param.data = self.tau * old_weight + (1 - self.tau) * update_weights

    def update_tau(self, step):
        self.tau = 1 - (1 - self.tau_base) * ((np.cos((np.pi * step) / self.total_steps) + 1) / 2)

    def augment(self, x):
        view1 = self.aug1(x)
        view2 = self.aug2(x)
        return view1, view2

    def loss_fn(self, student_out, teacher_out):
        student_out = F.normalize(student_out, dim=-1, p=2)
        teacher_out = F.normalize(teacher_out, dim=-1, p=2)

        loss = 2 - 2 * (student_out * teacher_out).sum(dim=-1)
        return loss

    def forward_pass(self, x):
        view1, view2 = self.augment(x)

        student_out_1 = self.projection_head(self.student(view1))
        student_out_2 = self.projection_head(self.student(view2))

        with torch.no_grad():
            teacher_out1 = self.projection_head(self.teacher(view1))
            teacher_out2 = self.projection_head(self.teacher(view2))

        loss = self.loss_fn(student_out_1, teacher_out2) + self.loss_fn(student_out_2, teacher_out1)

        return loss.mean()

    def backward_pass(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pretrain(self):
        step_counter = 0
        for epoch in range(self.num_epochs):
            loop = tqdm(enumerate(self.dataloader), leave=False)

            if epoch % self.lin_evaluation_frequency == 0:
                self.linear_evaluation(epoch)

            for idx, (x, _) in loop:
                loop.set_description(
                    f"Epoch {epoch + 1} / {self.num_epochs} | Step {idx} / {len(self.dataloader)} | Tau: {self.tau:.4f}"
                )
                x = x.to(self.device)
                loss = self.forward_pass(x)
                self.backward_pass(loss)

                self.update_teacher()
                self.update_tau(step_counter)
                step_counter += 1

    def linear_evaluation(self, after_k_epochs):
        print(f"\nPerforming linear evaluation after {after_k_epochs} epochs!\n")
        logger = get_tb_logger(self.logdir, f"_linear_evaluation_{after_k_epochs}_epochs")
        linear_eval = LinearEvaluator(self.mock_student_output, self.n_classes, device=self.device)
        self.change_requires_grad(self.student, False)
        loop = tqdm(range(linear_eval.epochs), desc=f"Linear Eval Protocol after {after_k_epochs} epochs")
        for epoch in loop:
            train_loss, val_loss = 0, 0
            train_accuracy, val_accuracy = 0, 0
            linear_eval.train()
            for idx, (x, y) in enumerate(self.dataloader):
                loss, preds = linear_eval.forward_backward(x, y, self.student)

                # calculate the loss and the accuracy
                train_loss += loss.item()
                train_accuracy += (preds == y.to(self.device)).sum().item()

            linear_eval.eval()
            for idx, (x, y) in enumerate(self.val_dataloader):
                with torch.no_grad():
                    loss, preds = linear_eval.forward_backward(x, y, self.student, backward=False)

                # calculate the loss and the accuracy
                val_loss += loss.item()
                val_accuracy += (preds == y.to(self.device)).sum().item()

            logger.add_scalar("LinearEvalLoss/Train", train_loss / len(self.dataloader), global_step=epoch)
            logger.add_scalar(
                "LinearEvalAccuracy/Train", train_accuracy / len(self.dataloader.dataset), global_step=epoch
            )
            logger.add_scalar("LinearEvalLoss/Val", val_loss / len(self.val_dataloader), global_step=epoch)
            logger.add_scalar(
                "LinearEvalAccuracy/Val", val_accuracy / len(self.val_dataloader.dataset), global_step=epoch
            )

        self.change_requires_grad(self.student, True)
        logger.flush()
        logger.close()
