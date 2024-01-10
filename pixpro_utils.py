from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from karies.data.dataset import CariesDataset
from pixel_level_contrastive_learning.pixel_level_contrastive_learning import (
    NetWrapper,
    PixelCL,
)
from torch.utils.data import DataLoader


class MaskRCNNModelWrapper(nn.Module):
    def __init__(self, mask_rcnn_transform, mask_rcnn_backbone):
        super(MaskRCNNModelWrapper, self).__init__()
        self.transform = mask_rcnn_transform
        self.backbone = mask_rcnn_backbone

        self.extra_layer = nn.Sequential(nn.Conv2d(256, 256, (3, 3), padding="same"), nn.MaxPool2d(3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

        self.pixel_layer_cheat = mask_rcnn_backbone.fpn.layer_blocks[-1]

    def forward(self, x: torch.Tensor):
        # treat x as a normal tensor, so the transform wants it to be a list
        x = [i for i in x]

        x = self.transform(x)[0].tensors
        x = self.backbone(x)["pool"]

        x = self.extra_layer(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class UNetModelWrapper(nn.Module):
    def __init__(self, encoder):
        super(UNetModelWrapper, self).__init__()
        self.encoder = encoder

        self.pixel_layer = encoder.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = x[:, 0:1]
        x = self.encoder(x)[-1]
        x = self.avgpool(x)
        x = self.flatten(x)

        return x


class PretrainingDataset(CariesDataset):
    def __init__(
        self,
        config,
        mode,
        dataset,
        in_channels,
        only_seg_masks: bool = False,
        crop_size: int = 112,
        unlabelled: Optional[str] = None,
    ):
        assert dataset in ["4k", "test"], "The dataset can be only 4k or test"
        assert unlabelled is None or unlabelled in ["bw", "bw+"], "Unlabelled can be None or one between ['bw', 'bw+]"
        self.config = config
        self.mode = mode
        self.only_seg_masks = only_seg_masks
        self.in_channels = in_channels

        data4k = Path("/cluster/group/karies_2022/Data/dataset_4k/images/")
        unlabelled_bw = Path("/cluster/group/karies_2022/Data/unlabelled_images/BW für FH Jan24")
        einzelzahnbilder = Path("/cluster/group/karies_2022/Data/unlabelled_images/Einzelzahnbilder")

        self.img_paths = []

        for paket in data4k.glob("*"):
            self.img_paths.extend(list((data4k / paket).glob("*")))
            if dataset == "test":
                break

        if unlabelled == "bw+":
            self.img_paths.extend(list(einzelzahnbilder.glob("*")))
        if unlabelled in ["bw", "bw+"]:
            self.img_paths.extend(list(unlabelled_bw.glob("*")))

        if self.config["image_shape"] is not None:
            self.img_resize = T.Resize((self.config["image_shape"][0], self.config["image_shape"][1]), antialias=True)
            self.mask_resize = T.Resize((self.config["image_shape"][0], self.config["image_shape"][1]), antialias=None)

        self.crop_size = crop_size

        image_transform = [
            T.ToTensor(),
            T.RandomCrop(self.crop_size),
        ]
        if self.config["image_shape"] is not None:
            image_transform.insert(
                1, T.Resize((self.config["image_shape"][0], self.config["image_shape"][1]), antialias=True)
            )

        self.image_transform = T.Compose(image_transform)

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item from dataset.

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: image tensor and label dict with boxes, labels and masks
        """

        image = cv2.imread(str(self.img_paths[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.image_transform(image)

        if self.in_channels == 1:
            image_tensor = image_tensor[0:1]

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


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))
