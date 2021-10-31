import torch
from sys import argv
from matplotlib import pyplot as plt
from os import path
from model import Autoencoder
from dataset import test_dataset

num_examples = 4
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
    model = Autoencoder(input_shape=input_shape, bottleneck_size=10)
    model.load_state_dict(model_state)

fig, axs = plt.subplots(2, num_examples, figsize=(num_examples*4, 2))

for i in range(num_examples):
    sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
    img, _ = test_dataset[sample_idx]
    img_t = img[None]
    y = model(img_t)
    axs[0, i].imshow(img[0], cmap="gray")

    y_img = y.detach().numpy()
    axs[1, i].imshow(y_img[0, 0], cmap="gray")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

axs[0, 0].set_ylabel('original', rotation=0, fontsize=15, labelpad=60)
axs[1, 0].set_ylabel('generated', rotation=0, fontsize=15, labelpad=70)
# plt.setp(axs[0, 0], ylabel='original')
# plt.setp(axs[1, 0], ylabel='generated')
# axs[0, 0].setp("Ground truth", rotation='vertical')
# axs[1, 0].setp("Generated", rotation='vertical')

plt.show()
