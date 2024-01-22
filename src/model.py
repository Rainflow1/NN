import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
import torch.nn as nn
import numpy as np


class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.conv2 = Down(64, 128)
        self.conv3 = Down(128, 256)
        self.conv4 = Down(256, 512)

        self.upConv1 = Up(512, 256)
        self.upConv2 = Up(256, 128)
        self.upConv3 = Up(128, 64)

        self.outConv = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv1 = self.upConv1(conv4, conv3)
        upconv2 = self.upConv2(upconv1, conv2)
        upconv3 = self.upConv3(upconv2, conv1)

        out = self.outConv(upconv3)
        return torch.sigmoid(out)
    
    pass


class Down(nn.Module):
    def __init__(self, input, output):
        super().__init__()

        self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(input, output, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output, output, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output, output, 3, 1, 1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

    pass

class Up(nn.Module):
    def __init__(self, input, output):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(input, output, 3, 2)
        self.conv = nn.Sequential(
                nn.Conv2d(input, output, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output, output, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(output, output, 3, 1, 1),
                nn.ReLU()
            )

    def forward(self, x, y):
        x = T.center_crop(self.upconv(x), y.size()[2:4])
        com = torch.cat([y, x], dim=1)
        return self.conv(com)

    pass