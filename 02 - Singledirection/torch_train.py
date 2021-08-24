import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import cv2
from PIL import Image
from resmodel import resnet18
from numpy.lib.stride_tricks import as_strided

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

EPOCH = 100
BATCH_SIZE = 64
LR = 0.01
SIZE = 256

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


def Calsim(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m-1, n))
    for i in range(m-1):
        for j in range(n):
            sim = 1 - abs(x[i, j] - x[i+1, j])
            # lf = 1 - abs(x[i+1, j+1] - x[i+1,j])
            # rt = 1 - abs(x[i+1, j+1] - x[i+1,j+2])
            # up = 1 - abs(x[i+1, j+1] - x[i, j+1])
            # dw = 1 - abs(x[i+1, j+1] - x[i+2, j+1])
            # if sim < 1:
            #     ret[i,j] = 0
            if sim < 1:
                ret[i,j] = 0
    # print(np.around(ret,2))
    return ret


class DealDataset(Dataset):
    def __init__(self, loader=default_loader):
        self.len = 10000
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
            fake_mask = Calsim(fake_mask1)
            fake_mask = np.expand_dims(fake_mask, 0)
            # print(np.around(fake_mask, 2))
            # print('\n')
            # print('\n')
            # print(fake_mask,'\n',np.around(fake_mask1,2))
            mask = torch.from_numpy(fake_mask)
            mask = torch.tensor(mask, dtype=torch.float32)
            label = torch.ones(1)
        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)

            mask = torch.ones((1, 15, 16), dtype=torch.float32)
            label = torch.zeros(1)
        # print(mask.shape)
        return img, mask

    def __len__(self):
        return self.len


# model=models.resnet18(pretrained=True)
# class_num = 1
# channel_in = model.fc.in_features
# model.fc = nn.Linear(channel_in,class_num)
# net = model.to(device)


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
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            print(np.around(torch.sigmoid(outputs).detach().cpu().numpy()[0],1))

            # print(np.around(labels.detach().cpu().numpy()[0],1))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # output = (outputs > 0.5).float()
            # correct = (output == labels).float().sum()

            losss = torch.mean((outputs - labels) ** 2)
            print('[epoch:%d, iter:%d] Loss: %.03f | MSE: %.3f ' % (epoch + 1, i, loss.item(), losss))
        torch.save(net, 'model.pkl')

            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, i, loss.item(), 100. * correct / BATCH_SIZE))
