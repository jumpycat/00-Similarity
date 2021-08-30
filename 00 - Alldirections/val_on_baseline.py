import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision

EPOCH = 10
BATCH_SIZE = 64
LR = 0.01
SIZE = 256
LENGTH = 10000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Deepfakes Face2Face FaceSwap NeuralTextures
train_fake_root = r'H:\FF++_Images_v2\Face2Face\raw\train/'

val_fake_root = r'H:\FF++_Images_v2\Face2Face\raw\val/'
net = torch.load(r'trained_models\v2\fs-resnet18_v2.pkl')
net.eval()

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
        fake_root = train_fake_root
        train_fake_video_paths = os.listdir(fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'H:\FF++_Images_v2\Real\raw\train/'
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
    # net = torchvision.models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512, 1)
    net = torch.load(r'trained_models\epoch-002-loss-0.006.pkl')

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
            # with open('logs.txt', 'a', encoding='utf-8') as f:
            #     f.write(data)
            #     f.write('\n')
        tag = 'epoch-%03d-loss-%.03f' % (epoch + 1, loss.item())
        torch.save(net, 'trained_models/' + tag + '.pkl')


def test():
    real_root = r'H:\FF++_Images_v2\Real\raw\val'
    test_real_video_paths = os.listdir(real_root)
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = real_root + '/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    # Deepfakes Face2Face FaceSwap NeuralTextures
    fake_root = val_fake_root
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
        for cc in range(size):
            if np.random.randint(0, 2):
                video_index = np.random.randint(0, NUM_fake)
                img_index = np.random.randint(0, len(test_fake_imgs[video_index]))
                img_path = test_fake_imgs[video_index][img_index]
                img_fake = default_loader(img_path)
                imgs.append(img_fake)
                labels.append(0)
            else:
                video_index = np.random.randint(0, NUM_real)
                img_index = np.random.randint(0, len(test_real_imgs[video_index]))
                img_path = test_real_imgs[video_index][img_index]
                img_real = default_loader(img_path)
                imgs.append(img_real)
                labels.append(1)
        # print(labels)
        return torch.stack(imgs, dim=0), labels

    prd = []
    lal = []
    for kk in range(100):
        inputs, label = getValdata(32)
        input = inputs.cuda()
        output = torch.sigmoid(net(input))
        prd += list(output.detach().cpu().numpy())
        lal += label

    prd = np.squeeze(np.array(prd)>0.5)
    acc = np.sum(prd==(np.array(lal)))/3200
    print(acc)

if __name__ == '__main__':
    test()