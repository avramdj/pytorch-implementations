import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torchsummary import summary

from model import VAE
from dataset import train_loader
from util import save_model, get_device

epochs = 20
learning_rate = 1e-3
input_shape = (1, 28, 28)
z_size = 128

device = get_device()
print(f'Using device "{device}"')

model = VAE(input_shape=input_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

summary(model, input_shape)

for epoch in range(epochs):
    loss = 0
    bce_total = 0
    kl_divergence_total = 0
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs, mus, log_vars = model(imgs)
        kl_divergence = 0.5 * torch.sum(-1 - log_vars + mus.pow(2) + log_vars.exp())
        bce = nn.functional.binary_cross_entropy(outputs, imgs, reduction="sum")
        train_loss = bce + kl_divergence
        train_loss.backward()
        optimizer.step()
        bce_total += bce
        kl_divergence_total += kl_divergence
        loss += train_loss.item()

    loss = loss / len(train_loader)

    print(
        "epoch : {}/{}, loss = {:.6f}, bce = {:.6f}, kl_divergence = {:.6f}".format(
            epoch + 1, epochs, loss, bce_total, kl_divergence_total
        )
    )

save_model(model)
