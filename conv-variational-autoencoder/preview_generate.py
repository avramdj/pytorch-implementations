from numpy.core.fromnumeric import reshape
import torch
from sys import argv
from matplotlib import pyplot as plt
from os import path
from model import VAE
import torchvision
import numpy as np
from tqdm import tqdm

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

num_examples = 20
input_shape = (1, 28, 28)

model_path = path.join("models", "latest")

if len(argv) > 1:
    model_path = argv[1]
    try:
        tmp = int(argv[2])
        num_examples = min(10, tmp)            
    except:
        pass

model = None
with open(model_path, "rb") as f:
    model_state = torch.load(f)
    model = VAE(input_shape=input_shape)
    model.load_state_dict(model_state)


with torch.no_grad():

    image_recon = []
    latent_x = np.linspace(-1.5,1.5,20)
    latent_y = np.linspace(-1.5,1.5,20)
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
    for i, lx in tqdm(enumerate(latent_x)):
        for j, ly in enumerate(latent_y):
            mu = torch.Tensor([lx, ly])
            eps = torch.Tensor([-6., -6.])
            mu = mu[None]
            eps = eps[None]
            image_recon.append(model.decode(mu, eps)[0])

    fig = plt.figure(figsize=(10, 10))

    show_image(torchvision.utils.make_grid(image_recon[:400],20,5))
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])
    plt.show()
