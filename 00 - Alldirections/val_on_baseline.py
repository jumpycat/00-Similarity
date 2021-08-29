import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision

EPOCH = 30
BATCH_SIZE = 64
LR = 0.01
SIZE = 256
LENGTH = 10000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
        self.len = LENGTH
        self.loader = loader

        # Deepfakes Face2Face FaceSwap NeuralTextures
        fake_root = r'D:\DATA\FF++_Images\Face2Face\raw\train/'
        train_fake_video_paths = os.listdir(fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'D:\DATA\FF++_Images\Real_1.8\raw\train/'
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
            label = torch.tensor([0])
        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)
            label = torch.tensor([1])

        return img, label

    def __len__(self):
        return self.len

def train():
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 1)

    net.to(device)
    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    for epoch in range(EPOCH):
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
            data = '[epoch:%03d, iter:%03d] Loss: %.03f' % (epoch + 1, i, loss.item())
            print(data)
            with open('logs.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')
        tag = 'epoch-%03d-loss-%.03f' % (epoch + 1, loss.item())
        torch.save(net, 'trained_models/' + tag + '.pkl')

def getValdata(size):
    real_root = r'D:\DATA\FF++_Images\Real_1.8\raw\val'
    test_real_video_paths = os.listdir(real_root)
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = real_root + '/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    #Deepfakes Face2Face FaceSwap NeuralTextures
    fake_root = r'D:\DATA\FF++_Images\Deepfakes\raw\val'
    test_fake_video_paths = os.listdir(fake_root)
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = fake_root + '/' + i
        img = os.listdir(video_path)
        test_fake_imgs.append([video_path + '/' + j for j in img])

    NUM_fake = len(test_fake_imgs)
    NUM_real = len(test_real_imgs)

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

    return torch.stack(imgs, dim=0),labels

def test():
    net = torch.load(r'trained_models\resnet18-f2f_1.8.pkl')
    prd = []
    lal = []
    for i in range(100):
        inputs, label = getValdata(32)
        input = inputs.cuda()
        output = net(input)
        prd += list(output.detach().cpu().numpy())
        lal += label
    prd = np.array(prd)>0.5
    acc = np.sum(prd==np.expand_dims(np.array(lal),axis=1))/3200
    print(acc)

if __name__ == '__main__':
    test()