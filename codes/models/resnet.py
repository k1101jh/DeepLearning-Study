import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        f_x = self.layers(x)
        x = f_x + self.shortcut(x)
        x = self.relu(x)
        
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.channels = [3, 64, 128, 256, 512, num_classes]
        self.block1 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], (7, 7), 2, 3),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
        )
        
        self.block2 = self._make_layers(self.channels[1], self.channels[1], 1, 2)
        self.block3 = self._make_layers(self.channels[1], self.channels[2], 2, 2)
        self.block4 = self._make_layers(self.channels[2], self.channels[3], 2, 2)
        self.block5 = self._make_layers(self.channels[3], self.channels[4], 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels[4], self.channels[5])
        )
    
    @classmethod
    def _make_layers(cls, in_channels, out_channels, stride, num_layers):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x