import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PetResNet(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.0):
        super(PetResNet, self).__init__()

        self.res = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        num_filters = self.res.fc.in_features

        self.res.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(num_filters, num_classes)
        )

    def forward(self, x):
        return self.res(x)

class PetNet(nn.Module):
    def __init__(self, num_classes=37, use_batchnorm=False, dropout_p=0.0):
        super(PetNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
