import torch.nn.functional as F
import os, torch
from weights import init
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



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

        self.convfirst = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)


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
        x = self.convfirst(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])



net = resnet34()
# pretext_model = torch.load(r'C:\Users\jumpycat\.cache\torch\checkpoints/resnet18-5c106cde.pth')
# model2_dict = net.state_dict()
# state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
# state_dict.pop('fc.weight')
# state_dict.pop('fc.bias')
# model2_dict.update(state_dict)
# net.load_state_dict(model2_dict)

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
        out = torch.cat((x1,x),dim=1)
        return out


class MISLnet(nn.Module):
    def __init__(self):
        super(MISLnet, self).__init__()
        self.LHPF = LHPF()
        self.base = net
        # self.base.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.LHPF(x)
        x = self.base(x)
        return x

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor

class DealDataset(Dataset):
    def __init__(self, loader=default_loader):
        self.len = 6400
        self.loader = loader
        # Deepfakes Face2Face FaceSwap NeuralTextures
        fake_root = r'D:\DATA\FF++_Images\NT\c40\train/'
        train_fake_video_paths = os.listdir(fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'D:\DATA\FF++_Images\Real\c40\train/'
        train_real_video_paths = os.listdir(real_root)
        self.train_real_imgs = []
        for i in train_real_video_paths:
            video_path = real_root + i
            img = os.listdir(video_path)
            self.train_real_imgs.append([video_path + '/' + j for j in img])
        self.NUM_fake = len(self.train_fake_imgs)
        self.NUM_real = len(self.train_real_imgs)

    def __getitem__(self, index):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, self.NUM_fake)
            img_index = np.random.randint(0, len(self.train_fake_imgs[video_index]))
            img_path = self.train_fake_imgs[video_index][img_index]
            img = self.loader(img_path)
            label = torch.tensor((0,))

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)
            label = torch.tensor((1,))

        return img,label

    def __len__(self):
        return self.len


def train():
    # net = MISLnet().to(device)
    # net = torchvision.models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512,1)
    net.to(device)

    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset, batch_size=32, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(30):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            input, label = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            th_acc = np.array(np.array(torch.sigmoid(output).detach().cpu().numpy()) > 0.5, dtype=int)
            acc = np.sum(th_acc == np.array(labels)) / 32

            data = '[epoch:%03d, iter:%03d] Loss: %.03f Acc: %.03f' % (epoch + 1, i, loss.item(), acc)
            print(data)
            # with open('logs-nt-c40-lsrm4.txt', 'a', encoding='utf-8') as f:
            #     f.write(data)
            #     f.write('\n')
        # tag = 'epoch-%03d-loss-%.03f-Acc-%.03f' % (epoch + 1, loss.item(), acc)
        # torch.save(net, 'models/nt-c40-lsrm/' + tag + '.pkl')

if __name__ == "__main__":
    # net = torch.load('models/epoch-002-loss-0.009-Acc-0.750.pkl')
    # print(list(net.parameters())[0])
    train()
    print('done!')