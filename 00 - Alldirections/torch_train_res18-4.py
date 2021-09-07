import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import torch.nn as nn
from torchsummary import summary

AVG_SIZE = 2
EPOCH = 100
BATCH_SIZE = 64
LR = 0.01
SIZE = 256
LENGTH = 10000
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



preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

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


def Calsimup(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m-1, n))
    for i in range(m-1):
        for j in range(n):
            if abs(x[i+1, j] - x[i, j]) > 0:
                ret[i,j] = 0
    return ret

def Calsimleft(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n-1))
    for i in range(m):
        for j in range(n-1):
            if abs(x[i, j+1] - x[i, j]) > 0:
                ret[i,j] = 0
    return ret


class DealDataset(Dataset):
    def __init__(self, loader=default_loader):
        self.len = LENGTH
        self.loader = loader
        # Deepfakes Face2Face FaceSwap NeuralTextures

        fake_root = r'D:\DATA\FF++_Images\Face2Face\c23\train/'
        train_fake_video_paths = os.listdir(fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'D:\DATA\FF++_Images\Real\c23\train/'
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

            mask_path = img_path.replace('c23','mask')

            fake_mask = cv2.imread(mask_path, 0)
            fake_mask = np.array(cv2.resize(fake_mask, (SIZE, SIZE)) > 1, dtype=np.float64)
            fake_mask1 = pool2d(fake_mask, 16, 16)

            fake_mask_up = Calsimup(fake_mask1)
            fake_mask_left = Calsimleft(fake_mask1)
            fake_mask_up = torch.from_numpy(np.expand_dims(fake_mask_up, 0))
            fake_mask_left = torch.from_numpy(np.expand_dims(fake_mask_left, 0))

            mask_up = torch.tensor(fake_mask_up, dtype=torch.float32)
            mask_left = torch.tensor(fake_mask_left, dtype=torch.float32)

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)

            mask_up = torch.ones((1, 15, 16), dtype=torch.float32)
            mask_left = torch.ones((1, 16, 15), dtype=torch.float32)

        return img, (mask_up,mask_up,mask_left,mask_left)

    def __len__(self):
        return self.len


def findthrehold(pred,label):
    best_acc = 0
    best_th = 0
    for th in [0.8 + mom/1000 for mom in range(200)]:
        threhold_acc = np.array(np.array(pred)>th,dtype=int)
        acc = np.sum(threhold_acc == np.array(label))/2000
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc,best_th


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

def val(model):
    model.eval()
    ret_hist = []
    ret_labels = []
    for i in range(80):
        inputs, label = getValdata(25)
        input = inputs.cuda()
        output1, output2,output3, output4 = model(input)

        up = torch.sigmoid(output1).detach().cpu().numpy()[:,:,:-1,1:15]
        down = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:,1:15]
        left = torch.sigmoid(output3).detach().cpu().numpy()[:,:,1:15,:-1]
        right = torch.sigmoid(output4).detach().cpu().numpy()[:,:,1:15,1:]

        sim_map = np.mean(np.concatenate((up, down, left, right), axis=1),axis=(1,2,3))
        batch_sim_map_avg = list(sim_map)

        ret_hist += batch_sim_map_avg
        ret_labels += label

    best_acc,best_th = findthrehold(ret_hist, ret_labels)
    return best_acc,best_th

real_root = r'D:\DATA\FF++_Images\Real\c23\val'
test_real_video_paths = os.listdir(real_root)
test_real_imgs = []
for i in test_real_video_paths:
    video_path = real_root + '/' + i
    img = os.listdir(video_path)
    test_real_imgs.append([video_path + '/' + j for j in img])

# Deepfakes Face2Face FaceSwap NeuralTextures
fake_root = r'D:\DATA\FF++_Images\Face2Face\c23\val/'
test_fake_video_paths = os.listdir(fake_root)
test_fake_imgs = []
for i in test_fake_video_paths:
    video_path = fake_root + '/' + i
    img = os.listdir(video_path)
    test_fake_imgs.append([video_path + '/' + j for j in img])

NUM_fake = len(test_fake_imgs)
NUM_real = len(test_real_imgs)




net = resnet18().to(device)
pretext_model = torch.load(r'C:\Users\jumpycat\.cache\torch\checkpoints/resnet18-5c106cde.pth')
model2_dict = net.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
state_dict.pop('fc.weight')
state_dict.pop('fc.bias')
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)

net.to(device)
dealDataset = DealDataset()
train_loader = DataLoader(dataset=dealDataset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

if __name__ == '__main__':
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, label1,label2, label3,label4 = inputs.to(device), labels[0].to(device), labels[1].to(device),\
                                    labels[2].to(device), labels[3].to(device)
            optimizer.zero_grad()
            output1, output2,output3, output4 = net(inputs)

            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            loss3 = criterion(output3, label3)
            loss4 = criterion(output4, label4)

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()


            data = '[epoch:%03d, iter:%03d] Loss: %.03f' % (epoch + 1, i, loss.item())
            print(data)
            with open('logs-c23-f2f-4.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')

        best_acc,best_th = val(net)


        tag = 'c23-f2f-4-epoch-%03d-loss-%.03f-ValAcc-%.03f-Threshold-%.03f' % (epoch + 1,loss.item(),best_acc,best_th)
        print(tag)
        torch.save(net, r'trained_models\v3\c23/'+tag + '.pkl')
