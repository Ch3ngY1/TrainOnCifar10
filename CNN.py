import torch.nn as nn
import torch
import math


class model(nn.Module):

    def __init__(self, input_sizes=32, intput_channels=3, classnum=10):
        super(model, self).__init__()
        self.input_sizes = input_sizes
        self.intput_channels = intput_channels
        self.class_out = classnum
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Conv2d(self.intput_channels, 256,
                                kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.layer2 = nn.Conv2d(256, 256,
                                kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.layer3 = nn.Conv2d(256, 256,
                                kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.FC1 = nn.Linear(4 * 4 * 256, 1024)
        self.FC2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, self.class_out)
        self.fc = nn.Linear(3*32*32, 4000)
        self.relu = nn.ReLU
        self.out = nn.Linear(4000, 10)


    def forward(self, x):

        x = self.maxpool(self.layer1(x)) # 16
        x = self.maxpool(self.layer2(x)) # 8
        x = self.maxpool(self.layer3(x)) # 4

        x = x.view(-1, 256 * 4 * 4)
        x = self.FC1(x)
        x = self.FC2(x)
        # x = torch.where(torch.isinf(x), torch.full_like(x, 80), x)
        # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        x = self.classifier(x)

        return x

if __name__ == '__main__':
    model = model()
    input = torch.randn(1, 3, 32, 32)
    out = model(input)
    print(out.shape)