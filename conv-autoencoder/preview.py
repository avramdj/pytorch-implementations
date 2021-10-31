import torch
from sys import argv
from matplotlib import pyplot as plt

from model import Autoencoder
from dataset import test_dataset

num_examples = 4
input_shape = (1, 28, 28)

if len(argv) < 2:
    model_path = argv[1]
    try:
        tmp = int(argv[2])
        num_examples = min(10, tmp)            
    except:
        pass

model = None
with open(model_path, "rb") as f:
    model_state = torch.load(f)
    model = Autoencoder(input_shape=input_shape, bottleneck_size=10)
    model.load_state_dict(model_state)

fig, axs = plt.subplots(num_examples, 2, figsize=(2, num_examples*4))

for i in range(num_examples):
    sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
    img, _ = test_dataset[sample_idx]
    img_t = img[None]
    y = model(img_t)
    axs[i, 0].imshow(img[0], cmap="gray")

    y_img = y.detach().numpy()
    axs[i, 1].imshow(y_img[0, 0], cmap="gray")

for ax in axs.flat:
    ax.axis("off")

axs[0, 0].set_title("Ground truth")
axs[0, 1].set_title("Generated")

plt.show()
