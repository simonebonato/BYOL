import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentations_byol import ByolAugmentations
from helper_classes import MLP, LinearEvaluator, StudentWrapper
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import get_tb_logger


class BYOL(nn.Module):
    def __init__(
        self,
        pretraining_model,
        dataloader,
        val_dataloader,
        num_epochs,
        logdir,
        lin_evaluation_frequency: Optional[int] = 10,
        tau_base=0.99,
        input_dims=1,
        n_classes=10,
        img_dims=(768, 1024),
        device="mps",
        hidden_features=2048,
        proj_output_features=256,
        byol_lr=3e-4,
        lin_eval_epochs: int = 100,
        mixed_precision_enabled=False,
    ):
        super(BYOL, self).__init__()

        self.student = pretraining_model.to(device)

        # set the teacher as the copy of the student and freeze its weights
        self.teacher = copy.deepcopy(pretraining_model).to(device)
        self.change_requires_grad(self.teacher, False)

        # defining objects and variables for the training
        self.dataloader, self.val_dataloader = dataloader, val_dataloader
        self.num_epochs = num_epochs
        self.total_steps = len(dataloader) * num_epochs
        self.tau, self.tau_base = tau_base, tau_base
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.device = device
        self.logdir = logdir

        self.student_projection_head = self.get_MLP_projector(img_dims, proj_output_features, hidden_features)
        self.prediction_head = MLP(proj_output_features, proj_output_features, hidden_features, device)

        self.teacher_projection_head = copy.deepcopy(self.student_projection_head).to(device)

        self.wrapped_student = StudentWrapper(pretraining_model, self.student_projection_head, self.prediction_head)
        self.optimizer = torch.optim.Adam(self.wrapped_student.parameters(), lr=byol_lr)

        self.aug1, self.aug2 = ByolAugmentations(img_dims).get_augmentations()

        self.lin_evaluation_frequency = lin_evaluation_frequency
        self.lin_eval_epochs = lin_eval_epochs

        self.mixed_precision_enabled = mixed_precision_enabled

    def batch_to_device(self, batch, device=None):
        to_device = self.device if device is None else device
        x, y = batch
        return x.to(to_device), y.to(to_device)

    def change_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def get_MLP_projector(self, img_dims, proj_output_features, hidden_features):
        mock_input = torch.randn(2, self.input_dims, *img_dims).to(self.device)
        mock_output = self.student(mock_input)

        self.mock_student_output = mock_output
        return MLP(mock_output.shape[-1], proj_output_features, hidden_features, self.device)

    def EMA_update(self, teacher_model, student_model):
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
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

        student_pred_1 = self.wrapped_student(view1)
        student_pred_2 = self.wrapped_student(view2)

        with torch.no_grad():
            teacher_proj_1 = self.teacher_projection_head(self.teacher(view1))
            teacher_proj_2 = self.teacher_projection_head(self.teacher(view2))

            teacher_proj_1.detach_()
            teacher_proj_2.detach_()

        loss = self.loss_fn(student_pred_1, teacher_proj_2) + self.loss_fn(student_pred_2, teacher_proj_1)

        return loss.mean()

    def backward_pass(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def pretrain(self):
        step_counter = 0
        self.scaler = GradScaler(enabled=self.mixed_precision_enabled)
        for epoch in range(self.num_epochs):
            loop = tqdm(enumerate(self.dataloader), leave=False)

            if isinstance(self.lin_evaluation_frequency, int) and epoch % self.lin_evaluation_frequency == 0:
                self.linear_evaluation(epoch)

            for idx, batch in loop:
                loop.set_description(
                    f"Epoch {epoch + 1} / {self.num_epochs} | Step {idx} / {len(self.dataloader)} | Tau: {self.tau:.4f}"
                )
                x, _ = self.batch_to_device(batch)

                with autocast(enabled=self.mixed_precision_enabled):
                    loss = self.forward_pass(x)
                    self.backward_pass(loss)

                self.EMA_update(self.teacher, self.student)
                self.EMA_update(self.teacher_projection_head, self.student_projection_head)

                self.update_tau(step_counter)
                step_counter += 1

    def linear_evaluation(self, after_k_epochs):
        print(f"\nPerforming linear evaluation after {after_k_epochs} epochs!\n")
        logger = get_tb_logger(self.logdir, f"_linear_evaluation_{after_k_epochs}_epochs")
        linear_eval = LinearEvaluator(
            self.mock_student_output, self.n_classes, device=self.device, epochs=self.lin_eval_epochs
        )
        self.change_requires_grad(self.student, False)
        loop = tqdm(range(linear_eval.epochs), desc=f"Linear Eval Protocol after {after_k_epochs} epochs")
        for epoch in loop:
            train_loss, val_loss = 0, 0
            train_accuracy, val_accuracy = 0, 0
            linear_eval.train()
            for idx, batch in enumerate(self.dataloader):
                x, y = self.batch_to_device(batch)
                loss, preds = linear_eval.forward_backward(x, y, self.student)

                # calculate the loss and the accuracy
                train_loss += loss.item()
                train_accuracy += (preds == y).sum().item()

            linear_eval.eval()
            for idx, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    x, y = self.batch_to_device(batch)
                    loss, preds = linear_eval.forward_backward(x, y, self.student, backward=False)

                # calculate the loss and the accuracy
                val_loss += loss.item()
                val_accuracy += (preds == y).sum().item()

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
