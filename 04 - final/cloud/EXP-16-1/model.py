import torch.nn as nn
import torch
from function import LocalAttention

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

        self.avgpl = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.f = nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1), bias=True)

        self.calsim_up = nn.Conv2d(256, 1, kernel_size=(2, 1), stride=1, bias=True)
        self.calsim_left = nn.Conv2d(256, 1, kernel_size=(1, 2), stride=1, bias=True)
        self.calsim_up_bank = nn.Conv2d(256, 1, kernel_size=(2, 1), stride=1, dilation=2, padding=1, bias=True)
        self.calsim_left_bank = nn.Conv2d(256, 1, kernel_size=(1, 2), stride=1, dilation=2, padding=1, bias=True)

        self.localattention_level1 = LocalAttention(inp_channels=64, out_channels=64, kH=8, kW=8)
        self.localattention_level2 = LocalAttention(inp_channels=128, out_channels=128, kH=4, kW=4)
        self.localattention_level3 = LocalAttention(inp_channels=256, out_channels=256, kH=2, kW=2)
        self.gamma1 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.gamma3 = nn.Parameter(torch.zeros(1),requires_grad=True)

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
        at1 = self.localattention_level1(x)
        x = self.gamma1 * at1 + x

        x = self.layer2(x)
        at2 = self.localattention_level2(x)
        x = self.gamma2 * at2 + x


        x = self.layer3(x)
        at3 = self.localattention_level3(x)
        x = self.gamma3 * at3 + x
        x = self.avgpl(x)
        x = self.f(x)
        x = nn.LeakyReLU(negative_slope=0.1)(x)

        up = self.calsim_up(x)
        left = self.calsim_left(x)
        up_bank = self.calsim_up_bank(x)
        left_bank = self.calsim_left_bank(x)

        # up2 = self.calsim_up2(up)
        # left2 = self.calsim_left2(left)
        # up_bank2 = self.calsim_up_bank2(up_bank)
        # left_bank2 = self.calsim_left_bank2(left_bank)

        return up, up_bank, left, left_bank
            # ,up2, up_bank2, left2, left_bank2


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
