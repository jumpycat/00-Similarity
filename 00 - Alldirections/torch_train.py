import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import cv2
from PIL import Image
from Alldirections.resmodel import resnet18
from numpy.lib.stride_tricks import as_strided
from Alldirections.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
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

def Calsimdown(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m-1, n))
    for i in range(m-1):
        for j in range(n):
            if abs(x[i, j] - x[i+1, j]) > 0:
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

def Calsimright(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n-1))
    for i in range(m):
        for j in range(n-1):
            if abs(x[i, j] - x[i, j+1]) > 0:
                ret[i,j] = 0
    return ret


class DealDataset(Dataset):
    def __init__(self, loader=default_loader):
        self.len = LENGTH
        self.loader = loader
        dst_train_path = r'D:\DATA\DeepFake\train/'
        train_real_video_paths = os.listdir(dst_train_path + 'real/')
        self.train_real_imgs = []
        for i in train_real_video_paths:
            video_path = dst_train_path + 'real/' + i
            img = os.listdir(video_path)
            self.train_real_imgs.append([video_path + '/' + j for j in img])

        train_fake_video_paths = os.listdir(dst_train_path + 'fake/')
        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = dst_train_path + 'fake/' + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])
        self.NUM_fake = len(self.train_fake_imgs)
        self.NUM_real = len(self.train_real_imgs)

    def __getitem__(self, index):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, self.NUM_fake)
            img_index = np.random.randint(0, len(self.train_fake_imgs[video_index]))
            img_path = self.train_fake_imgs[video_index][img_index]
            img = self.loader(img_path)

            mask_video = img_path.split('/')[-2]
            mask_img = img_path.split('/')[-1]
            mask_path = r'D:\DATA\DeepFake\train\mask/' + str(mask_video) + '/' + str(mask_img)

            fake_mask = cv2.imread(mask_path, 0)
            fake_mask = np.array(cv2.resize(fake_mask, (SIZE, SIZE)) > 1, dtype=np.float64)
            fake_mask1 = pool2d(fake_mask, 16, 16)
            # print(np.around(fake_mask1, 2))
            fake_mask_up = Calsimup(fake_mask1)
            fake_mask_down = Calsimdown(fake_mask1)
            fake_mask_left = Calsimleft(fake_mask1)
            fake_mask_right = Calsimright(fake_mask1)

            fake_mask_up = torch.from_numpy(np.expand_dims(fake_mask_up, 0))
            # fake_mask_down = torch.from_numpy(np.expand_dims(fake_mask_down, 0))
            fake_mask_left = torch.from_numpy(np.expand_dims(fake_mask_left, 0))
            # fake_mask_right = torch.from_numpy(np.expand_dims(fake_mask_right, 0))

            fake_mask_up = torch.tensor(fake_mask_up, dtype=torch.float32)
            # fake_mask_down = torch.tensor(fake_mask_down, dtype=torch.float32)
            fake_mask_left = torch.tensor(fake_mask_left, dtype=torch.float32)
            # fake_mask_right = torch.tensor(fake_mask_right, dtype=torch.float32)

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)

            fake_mask_up = torch.ones((1, 15, 16), dtype=torch.float32)
            # fake_mask_down = torch.ones((1, 15, 16), dtype=torch.float32)
            fake_mask_left = torch.ones((1, 16, 15), dtype=torch.float32)
            # fake_mask_right = torch.ones((1, 16, 15), dtype=torch.float32)

        return img, (fake_mask_up,fake_mask_left)

    def __len__(self):
        return self.len



net = resnet18()
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
            inputs, label1,label2 = inputs.to(device), labels[0].to(device), labels[1].to(device)
            optimizer.zero_grad()
            output1, output2 = net(inputs)

            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            data = '[epoch:%03d, iter:%03d] Loss: %.03f' % (epoch + 1, i, loss.item())
            print(data)
            with open('logs.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')
        tag = 'epoch-%03d-loss-%.03f' % (epoch + 1,loss.item())
        torch.save(net, 'models/'+tag + '.pkl')
