import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dims):
        super(Model, self).__init__()

        self.input_dims = input_dims

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_dims, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = self.flatten(x)

        return x
