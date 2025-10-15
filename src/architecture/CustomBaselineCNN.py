import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBaselineCNN(nn.Module):
    """
    Simple yet strong baseline for image → regression tasks.
    - Uses 4 convolutional blocks with batch norm and dropout
    - Global Average Pooling before regression head
    - Outputs `num_outputs` continuous values
    """

    def __init__(self, num_outputs=3):
        super(CustomBaselineCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 56 → 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 28 → 14
            nn.Dropout(0.1)
        )

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)           # shape: (B, 256, 1, 1)
        x = x.view(x.size(0), -1) # flatten to (B, 256)
        x = self.regressor(x)
        return x
