# Convolutional variational autoencoder implementation

CVAE with 2 latent dimensions

### Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 14, 14]          160
├─Conv2d: 1-2                            [-1, 64, 7, 7]            8,256
├─Conv2d: 1-3                            [-1, 128, 3, 3]           73,856
├─Flatten: 1-4                           [-1, 1152]                --
├─Linear: 1-5                            [-1, 2]                   2,306
├─Linear: 1-6                            [-1, 2]                   2,306
├─Linear: 1-7                            [-1, 1152]                3,456
├─Unflatten: 1-8                         [-1, 128, 3, 3]           --
├─ConvTranspose2d: 1-9                   [-1, 64, 7, 7]            73,792
├─ConvTranspose2d: 1-10                  [-1, 32, 14, 14]          8,224
├─ConvTranspose2d: 1-11                  [-1, 1, 28, 28]           129
==========================================================================================
Total params: 172,485
Trainable params: 172,485
Non-trainable params: 0
Total mult-adds (M): 6.42
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.17
Params size (MB): 0.66
Estimated Total Size (MB): 0.83
==========================================================================================
```

### Samples

#### Latent space distribution
<img src="https://user-images.githubusercontent.com/48069158/139599947-39d4e935-ce25-46f8-9e26-5001f2e21f2e.png" width="600" height="600">

![Figure_4](https://user-images.githubusercontent.com/48069158/139599839-b66c3e7a-15af-483f-9eeb-6dc55f23c192.png)
