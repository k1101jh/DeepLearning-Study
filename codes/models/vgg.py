import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.channels = [3, 64, 128, 256, 512, 4096, num_classes]
        # self.layers = nn.Sequential(
        #     nn.Conv2d(self.channels[0], self.channels[1], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[1], self.channels[1], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(self.channels[1], self.channels[2], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[2], self.channels[2], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(self.channels[2], self.channels[3], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[3], self.channels[3], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[3], self.channels[3], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(self.channels[3], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[4], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[4], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(self.channels[4], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[4], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channels[4], self.channels[4], (3, 3), 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        # )
        self.layers = nn.Sequential(
            self._make_layers(self.channels[0], self.channels[1], (3, 3), 1, 1, 2, use_max_pooling=True),
            self._make_layers(self.channels[1], self.channels[2], (3, 3), 1, 1, 2, use_max_pooling=True),
            self._make_layers(self.channels[2], self.channels[3], (3, 3), 1, 1, 3, use_max_pooling=True),
            self._make_layers(self.channels[3], self.channels[4], (3, 3), 1, 1, 3, use_max_pooling=True),
            self._make_layers(self.channels[4], self.channels[4], (3, 3), 1, 1, 3, use_max_pooling=True),
            # self.make_layers(self.channels[4], self.channels[5], (7, 7), 1, 0, 1, use_max_pooling=False),
            # self.make_layers(self.channels[5], self.channels[5], (1, 1), 1, 0, 1, use_max_pooling=False),
            # self.make_layers(self.channels[5], self.channels[6], (1, 1), 1, 0, 1, use_max_pooling=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels[4] * 7 * 7, self.channels[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.channels[5], self.channels[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.channels[5], self.channels[6]),
        )
    
    @classmethod
    def _make_layers(cls, in_channels, out_channels, filter_size, stride, padding, num_layers, use_max_pooling=False):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, filter_size, stride=stride, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, filter_size, stride=stride, padding=padding))
            layers.append(nn.ReLU(inplace=True))
        if use_max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x