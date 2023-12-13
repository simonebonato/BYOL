import torch
import torch.nn as nn


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
