import numpy as np
import torch
import torchvision.transforms as T

class OptionalTransform:
    def __init__(self, transform, transform_params, p):
        self.transform = transform(**transform_params)
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            return self.transform(x)
        else:
            return x


class Clamp:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return torch.clamp(x, self.min, self.max)


class ByolAugmentations:
    def __init__(self, img_shape: tuple):
        base_transforms = [
            T.RandomResizedCrop(size=img_shape, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            Clamp(0, 1),
            OptionalTransform(
                T.ColorJitter,
                {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
        ]

        self.aug1 = T.Compose(
            base_transforms
            + [
                OptionalTransform(T.GaussianBlur, {"kernel_size": (3, 3), "sigma": (0.1, 2.0)}, p=1.0),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )

        self.aug2 = T.Compose(
            base_transforms
            + [
                OptionalTransform(T.GaussianBlur, {"kernel_size": (3, 3), "sigma": (0.1, 2.0)}, p=0.1),
                OptionalTransform(T.RandomSolarize, {"threshold": 0.5}, p=0.2),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ],
        )

    def get_augmentations(self):
        return self.aug1, self.aug2
