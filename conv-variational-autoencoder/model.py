import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), z_size=2) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.output_shape = self.input_shape
        self.z_size = z_size

        self.e_hidden1 = nn.Conv2d(self.in_channels, 32, 2, 2)
        self.e_hidden2 = nn.Conv2d(32, 64, 2, 2)
        self.e_hidden3 = nn.Conv2d(64, 128, 3, 2)
        self.e_flatten = nn.Flatten()
        self.e_fc1 = nn.Linear(128 * 3 * 3, self.z_size)
        self.e_fc2 = nn.Linear(128 * 3 * 3, self.z_size)
        self.d_lin = nn.Linear(self.z_size, 128 * 3 * 3)
        self.d_unflatten = nn.Unflatten(1, (128, 3, 3))
        self.d_hidden1 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.d_hidden2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d_hidden3 = nn.ConvTranspose2d(32, self.in_channels, 2, 2)

    def forward(self, input):
        mu, log_var = self.encode(input)
        x = self.decode(mu, log_var)
        return x, mu, log_var

    def encode(self, input):
        x = torch.rrelu(self.e_hidden1(input))
        x = torch.rrelu(self.e_hidden2(x))
        x = torch.rrelu(self.e_hidden3(x))
        x = torch.rrelu(self.e_flatten(x))
        mu = self.e_fc1(x)
        log_var = self.e_fc2(x)
        return mu, log_var

    def decode(self, mu, log_var):
        x = torch.rrelu(self.sample(mu, log_var))
        x = torch.rrelu(self.d_lin(x))
        x = torch.rrelu(self.d_unflatten(x))
        x = torch.rrelu(self.d_hidden1(x))
        x = torch.rrelu(self.d_hidden2(x))
        x = torch.sigmoid(self.d_hidden3(x))
        return x

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps
