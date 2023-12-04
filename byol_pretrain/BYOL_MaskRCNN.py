from pathlib import Path

import torch
import torch.nn as nn
from BYOL import BYOL
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class BYOL_MaskRCNN(BYOL):
    def __init__(self, *args, original_maskrcnn, save_weights_every, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_evaluation_frequency = None

        self.original_maskrcnn = original_maskrcnn
        self.save_weights_every = save_weights_every

        if save_weights_every is not None:
            self.save_model_path = Path("./saved_pretrained_models") / self.logdir.stem
            self.save_model_path.mkdir(parents=True, exist_ok=True)

    def batch_to_device(self, batch, device=None):
        if device is None:
            device = self.device
        for i in range(len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        for label_dict in batch[1]:
            label_dict["boxes"] = label_dict["boxes"].to(device)
            label_dict["labels"] = label_dict["labels"].to(device)
            label_dict["masks"] = label_dict["masks"].to(device)

        return batch[0], batch[1]

    def augment(self, x):
        view1 = [self.aug1(img) for img in x]
        view2 = [self.aug2(img) for img in x]

        return view1, view2

    def pretrain(self):
        step_counter = 0
        self.scaler = GradScaler(enabled=self.mixed_precision_enabled)
        for epoch in range(self.num_epochs):
            loop = tqdm(enumerate(self.dataloader), leave=False)

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

            if self.save_weights_every is not None and epoch % self.save_weights_every == 0:
                checkpoint = {
                    "model": self.original_maskrcnn.model.state_dict(),
                    "optimizer": None,
                    "lr_scheduler": None
                    }
                torch.save(checkpoint, self.save_model_path / f"pretrain_{epoch}_epochs.pth")


class MaskRCNNModelWrapper(torch.nn.Module):
    def __init__(self, mask_rcnn_transform, mask_rcnn_backbone):
        super(MaskRCNNModelWrapper, self).__init__()
        self.transform = mask_rcnn_transform
        self.backbone = mask_rcnn_backbone

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def maskrcnn_reshape_transform(self, x: torch.Tensor):
        target_size = x["pool"].size()[-2:]
        activations = []
        for key, value in x.items():
            activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode="bilinear"))
        activations = torch.cat(activations, axis=1)
        return activations

    def forward(self, x: torch.Tensor):
        x = self.transform(x)[0].tensors
        x = self.backbone(x)

        x = self.maskrcnn_reshape_transform(x)
        x = self.gap(x)
        x = self.flatten(x)

        return x
