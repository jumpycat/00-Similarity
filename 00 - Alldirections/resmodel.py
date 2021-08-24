import torch
import torch.nn as nn
import numpy as np

from Alldirections.config import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc = nn.Linear(512 * block.expansion, 1)

        self.avgpl = nn.AvgPool2d((AVG_SIZE, AVG_SIZE), stride=(AVG_SIZE, AVG_SIZE))

        self.calsim_up = nn.Conv2d(256, 1, kernel_size=(2,1), stride=1, bias=True)
        self.calsim_down = nn.Conv2d(256, 1, kernel_size=(2,1), stride=1, bias=True)
        self.calsim_left = nn.Conv2d(256, 1, kernel_size=(1,2), stride=1, bias=True)
        self.calsim_right = nn.Conv2d(256, 1, kernel_size=(1,2), stride=1, bias=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpl(x)
        up = self.calsim_up(x)
        # down = self.calsim_down(x)
        left = self.calsim_left(x)
        # right = self.calsim_right(x)
        return up,left


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


if __name__ == '__main__':
    net = resnet18()
    print(resnet18())
'''
34 BasicBlock [3, 4, 6, 3]
50 Bottleneck [3, 4, 6, 3]
101 Bottleneck [3, 4, 23, 3]
152 Bottleneck [3, 4, 23, 3]
'''
