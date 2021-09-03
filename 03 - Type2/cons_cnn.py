import torch, torch.nn as nn, torch.nn.functional as F, random
import os, torch, random
from weights import init
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
        x = F.conv2d(x, self.weight.cuda(device),padding=2)
        return x


class MISLnet(nn.Module):
    def __init__(self):
        super(MISLnet, self).__init__()
        self.LHPF = LHPF()
        self.base = torchvision.models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(512, 1)

    def forward(self, x):
        # x = self.LHPF(x)
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
        self.len = 2700
        self.loader = loader
        # Deepfakes Face2Face FaceSwap NeuralTextures
        fake_root = r'I:\01-Dataset\01-Images\00-FF++\FaceShifter\raw\train/'
        train_fake_video_paths = os.listdir(fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'I:\01-Dataset\01-Images\00-FF++\Real\raw\train/'
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
    net = MISLnet().to(device)
    dealDataset = DealDataset()
    train_loader = DataLoader(dataset=dealDataset, batch_size=64, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
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

            th_acc = np.array(np.array(output.detach().cpu().numpy()) > 0.5, dtype=int)
            acc = np.sum(th_acc == np.array(labels)) / 64

            data = '[epoch:%03d, iter:%03d] Loss: %.03f Acc: %.03f' % (epoch + 1, i, loss.item(), acc)
            print(data)
            with open('logs-resnet18.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')
        tag = 'epoch-%03d-loss-%.03f-Acc-%.03f' % (epoch + 1, loss.item(), acc)
        torch.save(net, 'models/' + tag + '.pkl')

if __name__ == "__main__":
    # net = torch.load('models/epoch-002-loss-0.009-Acc-0.750.pkl')
    # print(list(net.parameters())[0])
    train()
    print('done!')