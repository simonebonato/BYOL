from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .augmentations_byol import ByolAugmentations
from .BYOL import BYOL_Class


class BYOL_UNet_Class(BYOL_Class):
    def __init__(self, *args, original_unet, save_weights_every, img_dims=1, **kwargs):
        super(BYOL_UNet_Class, self).__init__(*args, **kwargs)

        self.lin_evaluation_frequency = None
        self.val_dataloader = None

        self.original_unet = original_unet
        self.save_weights_every = save_weights_every

        self.aug1, self.aug2 = ByolAugmentations(img_dims, norm_mean=[0.5302], norm_std=[0.2247]).get_augmentations()

        if save_weights_every is not None:
            self.save_model_path = Path("./saved_pretrained_models/unet/") / self.logdir.stem
            self.save_model_path.mkdir(parents=True, exist_ok=True)

    def batch_to_device(self, batch, device=None):
        to_device = self.device if device is None else device
        x, y, _ = batch
        return x.to(to_device), None

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
                    "model": self.original_unet.model.state_dict(),
                    "optimizer": None,
                    "lr_scheduler": None,
                }
                torch.save(checkpoint, self.save_model_path / f"pretrain_{epoch}_epochs.pth")


class UNetModelWrapper(torch.nn.Module):
    def __init__(self, unet_encoder):
        super(UNetModelWrapper, self).__init__()
        self.unet_encoder = unet_encoder

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def unet_reshape_transform(self, feature_maps: list):
        target_size = feature_maps[0].size()[-2:]
        interpolated_feature_maps = []

        for feature_map in feature_maps:
            if feature_map.size()[-2:] == target_size:
                interpolated_feature_maps.append(feature_map)
            else:
                interpolated_feature_map = F.interpolate(
                    feature_map, size=target_size, mode="bilinear", align_corners=True
                )
                interpolated_feature_maps.append(interpolated_feature_map)

        concatenated_features = torch.cat(interpolated_feature_maps, dim=1)

        return concatenated_features

    def forward(self, x: torch.Tensor):
        x = self.unet_encoder(x)
        x = self.unet_reshape_transform(x)
        x = self.gap(x)
        x = self.flatten(x)

        return x
