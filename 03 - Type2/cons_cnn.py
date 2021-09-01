import torch, torch.nn as nn, torch.nn.functional as F, random
import os, torch, random


class MISLnet(nn.Module):
    def __init__(self):
        super(MISLnet, self).__init__()
        self.const_weight = nn.Parameter(torch.randn(size=[3, 1, 5, 5]), requires_grad=True)
        # self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=4)
        # self.conv2 = nn.Conv2d(96, 64, 5, stride=1, padding=2)
        # self.conv3 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        # self.conv4 = nn.Conv2d(64, 128, 1, stride=1)
        # self.fc1 = nn.Linear(6272, 200)
        # self.fc2 = nn.Linear(200, 200)
        # self.fc3 = nn.Linear(200, conf.total_class)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(3):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0

    def forward(self, x):
        # Constrained-CNN
        self.normalized_F()
        x = F.conv2d(x, self.const_weight)
        # CNN
        # x = self.conv1(x)
        # x = self.max_pool(torch.tanh(x))
        # x = self.conv2(x)
        # x = self.max_pool(torch.tanh(x))
        # x = self.conv3(x)
        # x = self.max_pool(torch.tanh(x))
        # x = self.conv4(x)
        # x = self.avg_pool(torch.tanh(x))
        # # Fully Connected
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = torch.tanh(x)
        # x = self.fc2(x)
        # x = torch.tanh(x)
        # logist = self.fc3(x)
        # output = F.softmax(logist, dim=1)
        return x


if __name__ == "__main__":
    model = MISLnet().cuda()
    print(model)
    # from torchsummary import summary
    # summary(model, (1, 256, 256))