import torch
import torch.nn as nn

class CIFAR10CustomNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.025):
        super(CIFAR10CustomNet, self).__init__()

        # C1: 4 standard convs
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # C2: 7 alternating convs (dilated and normal)
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # C3: 6 Depthwise Separable convs
        self.dwconv = nn.Sequential(
            *[layer for _ in range(6) for layer in [
                nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
                nn.Conv2d(64, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]]
        )

        # C40: 2 convs, last with stride=2
        self.c40 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.c5 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.dwconv(x)
        x = self.c40(x)
        x = self.gap(x)
        x = self.c5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CIFAR10CustomNet().to('cuda' if torch.cuda.is_available() else 'cpu')

from torchsummary import summary
summary(model, input_size=(3, 32, 32))
