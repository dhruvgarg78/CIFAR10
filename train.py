# train.py

"""
Training script for CIFAR-10 using CIFAR10CustomNet and Albumentations.
Includes mixup, gradient clipping, LR scheduling, and per-class accuracy reporting.
"""

# === Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import json
from tqdm import tqdm

# Local imports
from augmentation import CIFAR10Albumentations, train_transform, classes, imshow_grid, mean, std
from model import CIFAR10CustomNet

# === Device Configuration ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# === DataLoaders ===
batch_size = 64

# Train Loader
trainset = CIFAR10Albumentations(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Test Loader (standard normalization only)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# === Visualize a Batch of Training Images ===
dataiter = iter(trainloader)
images, labels = next(dataiter)
grid = torchvision.utils.make_grid(images[:64], nrow=8)
imshow_grid(grid, mean, std)
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(64)))

# === Model, Optimizer, Scheduler, Loss ===
model = CIFAR10CustomNet().to(device)
summary(model, input_size=(3, 32, 32))

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)

steps_per_epoch = len(trainloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.025,
    steps_per_epoch=steps_per_epoch,
    epochs=70,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e3
)

criterion = nn.CrossEntropyLoss()

# === Mixup Function ===
def mixup_data(x, y, alpha=0.4):
    '''Compute mixup data. Returns mixed inputs, pairs of targets, and lambda.'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# === Training Loop ===
num_epochs = 70
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(trainloader)
    pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixup augmentation
        inputs, targets_a, targets_b, lam = mixup_data(images, labels)
        outputs = model(inputs)

        # Loss with Mixup
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=loss.item(), lr=f'{lr:.6f}')

    avg_train_loss = running_loss / len(trainloader)
    train_losses.append(avg_train_loss)

    # === Evaluation ===
    model.eval()
    correct = 0
    total = 0
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    acc = 100 * correct / total
    test_accuracies.append(acc)

    # === Print Epoch Summary ===
    print(f"\n=== Epoch [{epoch+1}/{num_epochs}] Summary ===")
    print(f"Train Loss: {avg_train_loss:.4f} | Test Accuracy: {acc:.2f}%")

    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"Class [{classes[i]}] Accuracy: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    print("============================================\n")

# === Plotting Results ===
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.show()

# === Save Model and Results ===
torch.save(model.state_dict(), 'cifar10_customnet_epoch70_acc.pth')

with open('train_losses.json', 'w') as f:
    json.dump(train_losses, f)

with open('test_accuracies.json', 'w') as f:
    json.dump(test_accuracies, f)
