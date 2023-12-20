import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from karies.data.dataset import CariesDataset
from typing import Dict, Tuple
import cv2
import numpy as np



class MaskRCNNModelWrapper(nn.Module):
    def __init__(self, mask_rcnn_transform, mask_rcnn_backbone):
        super(MaskRCNNModelWrapper, self).__init__()
        self.transform = mask_rcnn_transform
        self.backbone = mask_rcnn_backbone

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, x: torch.Tensor):
        # treat x as a normal tensor, so the transform wants it to be a list
        x = [i for i in x]

        x = self.transform(x)[0].tensors
        x = self.backbone(x)["pool"]
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class PretrainingDataset(CariesDataset):

    def __init__(
        self,
        config,
        mode,
        only_seg_masks: bool = False,
        crop_size: int= 112,
    ):
        super().__init__(config, mode, only_seg_masks)
        self.crop_size = crop_size
        self.config = config

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item from dataset.

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: image tensor and label dict with boxes, labels and masks
        """
        label: Dict = self.labels[index]
        image = cv2.imread(str(label["path"]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config["histogram_eq"]:
            image = (image[:, :, 0] * 255).astype(np.uint8)
            image = self.equalize_clahe_image(image)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, -1).repeat(3, -1)

        masks = torch.tensor(np.zeros_like(image))
        image_tensor, _ = self.apply_augmentations_image_and_masks(image, masks)

        return image_tensor
    
    def get_data_loaders(self):

        gen = torch.Generator()
        if self.config["fix_random_seed"] > -1:
            gen.manual_seed(self.config["fix_random_seed"])

        # create data loaders
        SSL_loader = DataLoader(
            self,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            worker_init_fn=self.config["fix_random_seed"] if self.config["fix_random_seed"] > -1 else None,
            collate_fn=None,
            generator=gen,
        )
        return SSL_loader
    