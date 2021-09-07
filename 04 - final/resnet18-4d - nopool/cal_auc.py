import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

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
        down = self.calsim_down(x)
        left = self.calsim_left(x)
        right = self.calsim_right(x)
        return up,down,left,right

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Deepfakes Face2Face FaceSwap NeuralTextures

Fake_root = r'I:\01-Dataset\01-Images\00-FF++\Face2Face\c23\val'
net = torch.load(r'trained_models\v3\c23\c23-f2f-4-epoch-048-loss-0.105-ValAcc-0.982-Threshold-0.890.pkl')
net.eval()
TH = 0.932
AVG_SIZE = 2

def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


def pool2d(A, kernel_size, stride):
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.mean(axis=(1, 2)).reshape(output_shape)


def Calsim(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m - 1, n))
    for i in range(m - 1):
        for j in range(n):
            up = 1 - abs(x[i, j] - x[i + 1, j])
            ret[i, j] = up
    return ret

def findthrehold(pred,label):
    best_acc = 0
    best_th = 0
    for th in [0.6 + mom/1000 for mom in range(400)]:
        threhold_acc = np.array(np.array(pred)>th,dtype=int)
        acc = np.sum(threhold_acc == np.array(label))/3200
        if acc > best_acc:
            best_acc = acc
            best_th = th
    print('Threshold:',best_th,'Accuracy:',best_acc)

def showHISTandMsk():
    real_root = r'I:\01-Dataset\01-Images\00-FF++\Real\c23\val'
    test_real_video_paths = os.listdir(real_root)
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = real_root + '/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    fake_root = Fake_root
    test_fake_video_paths = os.listdir(fake_root)
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = fake_root + '/' + i
        img = os.listdir(video_path)
        test_fake_imgs.append([video_path + '/' + j for j in img])

    NUM_fake = len(test_fake_imgs)
    NUM_real = len(test_real_imgs)

    def getValdata(size):
        imgs = []
        labels = []
        for i in range(size):
            if np.random.randint(0, 2):
                video_index = np.random.randint(0, NUM_fake)
                img_index = np.random.randint(0, len(test_fake_imgs[video_index]))
                img_path = test_fake_imgs[video_index][img_index]
                img = default_loader(img_path)
                imgs.append(img)
                labels.append(0)
            else:
                video_index = np.random.randint(0, NUM_real)
                img_index = np.random.randint(0, len(test_real_imgs[video_index]))
                img_path = test_real_imgs[video_index][img_index]
                img = default_loader(img_path)
                imgs.append(img)
                labels.append(1)

        return torch.stack(imgs, dim=0), labels

    ret_hist = []
    ret_labels = []
    for i in range(100):
        inputs, label = getValdata(32)
        input = inputs.cuda()
        output1, output2,output3, output4 = net(input)

        up = torch.sigmoid(output1).detach().cpu().numpy()[:,:,:-1,1:15]
        down = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:,1:15]
        left = torch.sigmoid(output3).detach().cpu().numpy()[:,:,1:15,:-1]
        right = torch.sigmoid(output4).detach().cpu().numpy()[:,:,1:15,1:]

        sim_map = np.mean(np.concatenate((up, down, left, right), axis=1),axis=(1,2,3))
        batch_sim_map_avg = list(sim_map)

        ret_hist += batch_sim_map_avg
        ret_labels += label

    findthrehold(ret_hist, ret_labels)


    plt.hist(ret_hist, bins=100)
    plt.xlabel('mean')
    plt.ylabel('num')
    plt.show()

    return ret_labels,ret_hist

def calcAUC_byProb(labels, probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []
    for _,i in enumerate(labels):
        if (i == 1):
            P += 1
            pos_prob.append(probs[_])
        else:
            N += 1
            neg_prob.append(probs[_])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if (pos > neg):
                number += 1
            elif (pos == neg):
                number += 0.5
    return number / (N * P)

if __name__ == '__main__':
    labels, probs = showHISTandMsk()
    auc = calcAUC_byProb(labels,probs)
    print(auc)





