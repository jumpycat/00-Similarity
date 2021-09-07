import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')

class efficient(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.l = nn.Sequential()
        self.l.add_module('conv_stem', m._conv_stem)
        self.l.add_module('_bn0', m._bn0)
        self.l.add_module('_blocks', nn.Sequential(*m._blocks)[:-5])
        self.calsim_up = nn.Conv2d(112, 1, kernel_size=(2, 1), stride=1, bias=True)
        self.calsim_down = nn.Conv2d(112, 1, kernel_size=(2, 1), stride=1, bias=True)
        self.calsim_left = nn.Conv2d(112, 1, kernel_size=(1, 2), stride=1, bias=True)
        self.calsim_right = nn.Conv2d(112, 1, kernel_size=(1, 2), stride=1, bias=True)

        self.calsim_up_bank = nn.Conv2d(112, 1, kernel_size=(2, 1), stride=1, dilation=2,padding=1,bias=True)
        self.calsim_down_bank = nn.Conv2d(112, 1, kernel_size=(2, 1), stride=1, dilation=2,padding=1,bias=True)
        self.calsim_left_bank = nn.Conv2d(112, 1, kernel_size=(1, 2), stride=1, dilation=2,padding=1,bias=True)
        self.calsim_right_bank = nn.Conv2d(112, 1, kernel_size=(1, 2), stride=1, dilation=2,padding=1,bias=True)


    def forward(self, x):
        x = self.l(x)
        up = self.calsim_up(x)
        down = self.calsim_down(x)
        left = self.calsim_left(x)
        right = self.calsim_right(x)

        up_bank = self.calsim_up_bank(x)
        down_bank = self.calsim_down_bank(x)
        left_bank = self.calsim_left_bank(x)
        right_bank = self.calsim_right_bank(x)

        return up, down, left, right,up_bank, down_bank, left_bank, right_bank



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# Deepfakes Face2Face FaceSwap NeuralTextures

Fake_root = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Face2Face/c23/val'
net = torch.load(r'models/df-epoch-015-loss-0.093-ValAcc-0.980-Threshold-0.880.pkl')
net.eval()

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
    for th in [0.8 + mom/1000 for mom in range(200)]:
        threhold_acc = np.array(np.array(pred)>th,dtype=int)
        acc = np.sum(threhold_acc == np.array(label))/5000
        if acc > best_acc:
            best_acc = acc
            best_th = th
            # print('Threshold:', best_th, 'Accuracy:', best_acc)
    print('Threshold:',best_th,'Accuracy:',best_acc)

def showHISTandMsk():
    real_root = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Real/c23/val'
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
        inputs, label = getValdata(50)
        input = inputs.cuda()
        output1, output2,output3, output4 = net(input)

        up = torch.sigmoid(output1).detach().cpu().numpy()[:,:,:-1,1:15]
        down = torch.sigmoid(output1).detach().cpu().numpy()[:,:,1:,1:15]
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
    plt.savefig("res/df->f2f-c23-8.png")
    # plt.show()

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





