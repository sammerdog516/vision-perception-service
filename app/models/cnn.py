import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # -> (32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # -> (64, 11, 11)

        self.pool = nn.MaxPool2d(2) # halves spatial dims

        # After conv + pool:
        # conv1: (32, 26, 26) -> pool -> (32, 13, 13)
        # conv2: (64, 11, 11) -> pool -> (64, 5, 5))
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)    # flatten
        x = self.fc(x)               # logits
        return x