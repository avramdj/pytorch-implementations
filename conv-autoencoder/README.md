# Convolutional autoencoder implementation

Convolutional AE with z = 10

### Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 32, 14, 14]          160
├─Conv2d: 1-2                            [-1, 64, 7, 7]            8,256
├─Conv2d: 1-3                            [-1, 128, 3, 3]           73,856
├─Flatten: 1-4                           [-1, 1152]                --
├─Linear: 1-5                            [-1, 10]                  11,530
├─Linear: 1-6                            [-1, 1152]                12,672
├─Unflatten: 1-7                         [-1, 128, 3, 3]           --
├─ConvTranspose2d: 1-8                   [-1, 64, 7, 7]            73,792
├─ConvTranspose2d: 1-9                   [-1, 32, 14, 14]          8,224
├─ConvTranspose2d: 1-10                  [-1, 1, 28, 28]           129
==========================================================================================
Total params: 188,619
Trainable params: 188,619
Non-trainable params: 0
Total mult-adds (M): 6.43
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.17
Params size (MB): 0.72
Estimated Total Size (MB): 0.89
==========================================================================================
```

### Sample

![Figure_1](https://user-images.githubusercontent.com/48069158/139596086-c8753ded-35b3-49d5-a09a-a91d0186d64c.png)
