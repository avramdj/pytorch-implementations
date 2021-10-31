import torch
from torch import nn
from torch import optim
from datetime import datetime
from os import path
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchsummary import summary

from model import Autoencoder
from dataset import train_loader
from util import save_model, get_device

epochs = 15
learning_rate = 1e-3
input_shape = (1, 28, 28)
bottleneck_size = 10

device = get_device()
print(f'Using device "{device}"')

model = Autoencoder(input_shape=input_shape, bottleneck_size=bottleneck_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

summary(model, input_shape)

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in tqdm(train_loader):
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(train_loader)

    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

save_model(model)