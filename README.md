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

## Results

- ✅ **Final test accuracy:** ~90.5%  
- ✅ **Parameter count:** <190k  
- ✅ **Converges within ~70 epochs**

## Techniques that enabled 99.5% under 190k params

- ✅ **Depthwise Separable Convolutions** — drastically reduced parameter count and computational cost
- ✅ **Dilation** — used in specific layers to increase effective receptive field
- ✅ **Mixup augmentation** — helped generalization and stabilized training
- ✅ **OneCycleLR scheduler** — allowed higher learning rates and fast convergence
- ✅ **Coarse Dropout (Cutout)** — improved robustness
- ✅ **Gradient Clipping** — avoided gradient explosions during aggressive learning phases

## Dependencies

- PyTorch
- Albumentations
- torchvision
- tqdm
- matplotlib
- opencv-python-headless

## License

MIT License
