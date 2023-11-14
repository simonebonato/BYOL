import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Model, self).__init__()
        

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_dims, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fcl = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_dims)
        )

        self.relu_pool = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()


    def forward(self, x):

        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fcl(x)

        return x
