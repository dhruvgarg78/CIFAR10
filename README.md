# CIFAR-10 Custom CNN

A high-performance CNN for the CIFAR-10 dataset, trained to achieve **90.5% test accuracy** with **<190k parameters**.

![Screenshot 2025-06-09 160953](https://github.com/user-attachments/assets/21a945f0-6873-4447-aff1-d8a61c5364f0)

![Screenshot 2025-06-09 161246](https://github.com/user-attachments/assets/a884e171-13d8-4f2c-98fc-f09d6d56c6f4)

## Key Features

- ⚡ **Albumentations-based data augmentation** (horizontal flip, coarse dropout, shift-scale-rotate)
- ⚡ **Depthwise Separable Convolutions** to reduce parameters and increase efficiency
- ⚡ **Dilation** in selected convolutional layers to increase receptive field without increasing parameters
- ⚡ **Mixup augmentation** for regularization and improved generalization
- ⚡ **OneCycle Learning Rate Scheduler** for faster convergence and improved performance
- ⚡ **Gradient Clipping** to stabilize training
- ⚡ **Per-class accuracy reporting** after each epoch

## Architecture

- 2 ConvBlock layers (standard)
- 4 alternating **dilated** + **depthwise separable** conv layers
- 4 depthwise separable conv layers
- Final block with stride and dilation
- Global Average Pooling
- Final FC layer

## Project Structure

```
augmentation.py                       # Albumentations transforms, dataset wrapper, visualization utilities
model.py                              # ConvBlock, DepthwiseSeparableConv, CIFAR10CustomNet architecture
train.py                              # Training loop, mixup, LR scheduler, evaluation, model saving
cifar10_customnet_epoch70_acc90.41.pth # Trained model weights
train_losses.json                     # Training loss log
test_accuracies.json                  # Test accuracy log
logs1.txt                             # Logs (user-generated)
requirements.txt                      # Dependencies
```

## Usage

```bash
# Clone repo
git clone https://github.com/dhruvgarg78/CIFAR10.git
cd CIFAR10

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py
```


## Receptive Field Calculations
```
| Block / Layer     | RF after layer |
| ----------------- | -------------- |
| Input             | 1              |
| Conv1             | 3              |
| Conv2             | 5              |
| Conv3 (dilated)   | 9              |
| Depthwise4        | 11             |
| Conv5 (dilated)   | 15             |
| Depthwise6        | 17             |
| Depthwise7        | 19             |
| Depthwise8        | 21             |
| Depthwise9        | 23             |
| Depthwise10       | 25             |
| Conv11            | 27             |
| Conv12 (stride=2) | 29             |
| Conv13 (dilated)  | 37             |
| Conv14 (dilated)  | 45             |
| Depthwise15       | 49             |
| GAP               | Full image RF  |
```

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
           Dropout-4           [-1, 32, 32, 32]               0
         ConvBlock-5           [-1, 32, 32, 32]               0
            Conv2d-6           [-1, 64, 32, 32]          18,432
       BatchNorm2d-7           [-1, 64, 32, 32]             128
              ReLU-8           [-1, 64, 32, 32]               0
           Dropout-9           [-1, 64, 32, 32]               0
        ConvBlock-10           [-1, 64, 32, 32]               0
           Conv2d-11           [-1, 64, 32, 32]          36,864
      BatchNorm2d-12           [-1, 64, 32, 32]             128
             ReLU-13           [-1, 64, 32, 32]               0
          Dropout-14           [-1, 64, 32, 32]               0
        ConvBlock-15           [-1, 64, 32, 32]               0
           Conv2d-16           [-1, 64, 32, 32]             576
           Conv2d-17           [-1, 64, 32, 32]           4,096
      BatchNorm2d-18           [-1, 64, 32, 32]             128
             ReLU-19           [-1, 64, 32, 32]               0
          Dropout-20           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-21           [-1, 64, 32, 32]               0
           Conv2d-22           [-1, 64, 32, 32]          36,864
      BatchNorm2d-23           [-1, 64, 32, 32]             128
             ReLU-24           [-1, 64, 32, 32]               0
          Dropout-25           [-1, 64, 32, 32]               0
        ConvBlock-26           [-1, 64, 32, 32]               0
           Conv2d-27           [-1, 64, 32, 32]             576
           Conv2d-28           [-1, 64, 32, 32]           4,096
      BatchNorm2d-29           [-1, 64, 32, 32]             128
             ReLU-30           [-1, 64, 32, 32]               0
          Dropout-31           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-32           [-1, 64, 32, 32]               0
           Conv2d-33           [-1, 64, 32, 32]             576
           Conv2d-34           [-1, 64, 32, 32]           4,096
      BatchNorm2d-35           [-1, 64, 32, 32]             128
             ReLU-36           [-1, 64, 32, 32]               0
          Dropout-37           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-38           [-1, 64, 32, 32]               0
           Conv2d-39           [-1, 64, 32, 32]             576
           Conv2d-40           [-1, 64, 32, 32]           4,096
      BatchNorm2d-41           [-1, 64, 32, 32]             128
             ReLU-42           [-1, 64, 32, 32]               0
          Dropout-43           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-44           [-1, 64, 32, 32]               0
           Conv2d-45           [-1, 64, 32, 32]             576
           Conv2d-46           [-1, 64, 32, 32]           4,096
      BatchNorm2d-47           [-1, 64, 32, 32]             128
             ReLU-48           [-1, 64, 32, 32]               0
          Dropout-49           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-50           [-1, 64, 32, 32]               0
           Conv2d-51           [-1, 64, 32, 32]             576
           Conv2d-52           [-1, 64, 32, 32]           4,096
      BatchNorm2d-53           [-1, 64, 32, 32]             128
             ReLU-54           [-1, 64, 32, 32]               0
          Dropout-55           [-1, 64, 32, 32]               0
DepthwiseSeparableConv-56           [-1, 64, 32, 32]               0
           Conv2d-57           [-1, 32, 32, 32]          18,432
      BatchNorm2d-58           [-1, 32, 32, 32]              64
             ReLU-59           [-1, 32, 32, 32]               0
          Dropout-60           [-1, 32, 32, 32]               0
        ConvBlock-61           [-1, 32, 32, 32]               0
           Conv2d-62           [-1, 32, 16, 16]           9,216
      BatchNorm2d-63           [-1, 32, 16, 16]              64
             ReLU-64           [-1, 32, 16, 16]               0
          Dropout-65           [-1, 32, 16, 16]               0
        ConvBlock-66           [-1, 32, 16, 16]               0
           Conv2d-67           [-1, 32, 14, 14]           9,216
      BatchNorm2d-68           [-1, 32, 14, 14]              64
             ReLU-69           [-1, 32, 14, 14]               0
        ConvBlock-70           [-1, 32, 14, 14]               0
           Conv2d-71           [-1, 64, 12, 12]          18,432
      BatchNorm2d-72           [-1, 64, 12, 12]             128
             ReLU-73           [-1, 64, 12, 12]               0
        ConvBlock-74           [-1, 64, 12, 12]               0
           Conv2d-75           [-1, 64, 12, 12]             576
           Conv2d-76           [-1, 64, 12, 12]           4,096
      BatchNorm2d-77           [-1, 64, 12, 12]             128
             ReLU-78           [-1, 64, 12, 12]               0
DepthwiseSeparableConv-79           [-1, 64, 12, 12]               0
AdaptiveAvgPool2d-80             [-1, 64, 1, 1]               0
           Conv2d-81             [-1, 64, 1, 1]           4,096
        ConvBlock-82             [-1, 64, 1, 1]               0
           Linear-83                   [-1, 10]             650
================================================================
Total params: 187,434
Trainable params: 187,434
Non-trainable params: 0
```
