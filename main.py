import torch
import torchvision
from torch import nn
from torch import optim
from datetime import datetime
from os import path
from tqdm import tqdm
from matplotlib import pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.input_shape = kwargs["input_shape"]
        self.output_shape = self.input_shape
        self.bottleneck_size = kwargs["bottleneck_size"]

        self.pool = nn.MaxPool2d(2, 2, padding=1)
        self.e_hidden1 = nn.Conv2d(1, 32, 2, 2)
        self.e_hidden2 = nn.Conv2d(32, 64, 2, 2)
        self.e_hidden3 = nn.Conv2d(64, 128, 3, 2)
        self.e_flatten = nn.Flatten()
        self.e_lin = nn.Linear(128 * 3 * 3, self.bottleneck_size)
        self.d_lin = nn.Linear(self.bottleneck_size, 128 * 3 * 3)
        self.d_unflatten = nn.Unflatten(1, (128, 3, 3))
        self.d_hidden1 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.d_hidden2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d_hidden3 = nn.ConvTranspose2d(32, 1, 2, 2)

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
        x = torch.rrelu(self.d_hidden3(x))
        return x


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2
)

epochs = 10
learning_rate = 1e-3
input_shape = (28, 28)
bottleneck_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device \"{device}\"")

model = Autoencoder(input_shape=input_shape, bottleneck_size=bottleneck_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

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

save_path = path.join("models", f"model-state-{datetime.now(tz=None)}")

torch.save(model.state_dict(), save_path)

sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
img, _ = test_dataset[sample_idx]
img_t = img[None].to(device)
y = model(img_t)

plt.imshow(img[0], cmap="gray")

y_img = y.detach().cpu().numpy()
plt.imshow(y_img[0, 0], cmap="gray")

embedding = model.encode(img.to(device)[None])
