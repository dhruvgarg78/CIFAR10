# augmentation.py

"""
Defines data augmentations using Albumentations and CIFAR10Albumentations dataset wrapper.
Provides utility function for image visualization.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

# CIFAR-10 mean and std values
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

# Albumentations transform for training images
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=mean,
        mask_fill_value=None,
        p=0.5
    ),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Albumentations wrapper for CIFAR-10 dataset
class CIFAR10Albumentations(CIFAR10):
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = np.array(image)

        if self.albumentations_transform:
            image = self.albumentations_transform(image=image)['image']

        return image, label

# Class labels
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Utility function to display grid of images
def imshow_grid(images, mean, std, nrow=8):
    # Unnormalize
    inv_trans = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    images = inv_trans(images)

    # Clamp to [0,1] for imshow
    images = torch.clamp(images, 0.0, 1.0)

    npimg = images.numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()
