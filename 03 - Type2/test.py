import numpy as np
import torch.nn.functional as F
import torch
from weights import init
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


base = EfficientNet.from_pretrained('efficientnet-b0')
feature = base._fc.in_features
base._fc = nn.Linear(in_features=feature,out_features=1,bias=True)


class LHPF(nn.Module):
    def __init__(self):
        super(LHPF, self).__init__()
        self.weight = nn.Parameter(data=init,requires_grad=True)
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        central_pixel = (self.weight.data[:, :, 2, 2])
        for i in range(3):
            for j in range(3):
                sumed = (self.weight.data[i][j].sum() - central_pixel[i][j])
                self.weight.data[i][j] /= sumed
                self.weight.data[i, j, 2, 2] = -1.0

    def forward(self, x):
        self.reset_parameters()
        x1 = F.conv2d(x, self.weight.cuda(device),padding=2)
        # out = torch.cat((x1,x),dim=1)
        return x1


class MISLnet(nn.Module):
    def __init__(self):
        super(MISLnet, self).__init__()
        self.LHPF = LHPF()
        self.base = base

    def forward(self, x):
        x = self.LHPF(x)
        x = self.base(x)
        return x


# model = torch.load(r'C:\Users\jumpycat\Desktop\00-YunshuDai\00-Similarity\03 - Type2\models\nt-c40-lsrm/EXP7 -  epoch-030-loss-0.122-Acc-0.969.pkl')
#
# weights = model.LHPF.weight.detach().cpu().numpy()
# print(weights.shape)
# for i in range(3):
#     for j in range(3):
#         print(np.sum(weights[i][j][3][3]))

inputs = torch.rand(1, 3, 224, 224)
model = EfficientNet.from_pretrained('efficientnet-b0')
endpoints = model.children()
# for i, module in enumerate(endpoints):
#     print(i, module)

new_model = nn.Sequential(*list(model.children())[:-6]).to(device)

summary(new_model,(3,256,256))