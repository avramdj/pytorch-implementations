import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.input_shape = kwargs["input_shape"]
        self.in_channels = self.input_shape[0]
        self.output_shape = self.input_shape
        self.bottleneck_size = kwargs["bottleneck_size"]

        self.pool = nn.MaxPool2d(2, 2, padding=1)
        self.e_hidden1 = nn.Conv2d(self.in_channels, 32, 2, 2)
        self.e_hidden2 = nn.Conv2d(32, 64, 2, 2)
        self.e_hidden3 = nn.Conv2d(64, 128, 3, 2)
        self.e_flatten = nn.Flatten()
        self.e_lin = nn.Linear(128 * 3 * 3, self.bottleneck_size)
        self.d_lin = nn.Linear(self.bottleneck_size, 128 * 3 * 3)
        self.d_unflatten = nn.Unflatten(1, (128, 3, 3))
        self.d_hidden1 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.d_hidden2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d_hidden3 = nn.ConvTranspose2d(32, self.in_channels, 2, 2)

    def forward(self, input):
        x = self.encode(input)
        x = self.decode(x)
        return x

    def encode(self, input):
        x = torch.rrelu(self.e_hidden1(input))
        x = torch.rrelu(self.e_hidden2(x))
        x = torch.rrelu(self.e_hidden3(x))
        x = torch.rrelu(self.e_flatten(x))
        x = torch.rrelu(self.e_lin(x))
        return x

    def decode(self, input):
        x = torch.rrelu(self.d_lin(input))
        x = torch.rrelu(self.d_unflatten(x))
        x = torch.rrelu(self.d_hidden1(x))
        x = torch.rrelu(self.d_hidden2(x))
        x = torch.relu(self.d_hidden3(x))
        return x
