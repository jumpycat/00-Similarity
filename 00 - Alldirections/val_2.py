import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SIZE = 256

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


def Calsim(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m - 1, n))
    for i in range(m - 1):
        for j in range(n):
            up = 1 - abs(x[i, j] - x[i + 1, j])
            ret[i, j] = up
    return ret


def getValdata(size):
    test_real_video_paths = os.listdir(r'D:\DATA\FF++_Images\Real\raw\val')
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = r'D:\DATA\FF++_Images\Real\raw\val/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    test_fake_video_paths = os.listdir(r'D:\DATA\FF++_Images\NeuralTextures\raw\val')
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = r'D:\DATA\FF++_Images\NeuralTextures\raw\val/' + i
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


net = torch.load(r'trained_models\epoch-012-loss-0.048.pkl')


def showHist():
    ret = []
    for i in range(10):
        inputs,_ = getValdata(32)
        input = inputs.cuda()
        output1,output2 = net(input)
        outputs = list(np.mean(torch.sigmoid(output1.detach().cpu()).numpy(),axis=(1,2,3)))
        # outputs = list(np.mean(output1.detach().cpu().numpy(),axis=(1,2,3)))
        ret += outputs

    print(ret)
    plt.hist(ret, bins=100)
    plt.xlabel('mean')
    plt.ylabel('num')
    plt.show()

def showMask():
    inputs,label = getValdata(32)
    input = inputs.cuda()
    output1, output2 = net(input)

    up = torch.sigmoid(output1).detach().cpu().numpy()[:,:,:-1,1:15]
    down = torch.sigmoid(output1).detach().cpu().numpy()[:,:,1:,1:15]
    left = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:15,:-1]
    right = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:15,1:]

    inputs = inputs.numpy()

    sim_map = np.mean(np.concatenate((up,down,left,right),axis=1),axis=1)
    print(sim_map.shape)
    for i in range(16):
        plt.subplot(4, 8, i + 1 + 8*(i//8))
        plt.imshow(sim_map[i]<0.5, cmap='gray')
        plt.xlabel(label[i])
        plt.subplot(4, 8, i + 1 + 8*(i//8+1))
        plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
    plt.show()

def calacc():
    ret = 0
    for i in range(10):
        inputs,labels = getValdata(32)
        input = inputs.cuda()
        output1,output2 = net(input)
        outputs = np.array(np.mean(torch.sigmoid(output1.detach().cpu()).numpy(),axis=(1,2,3))>0.9,dtype=int)
        res = np.sum(outputs==np.array(labels))
        ret += res
        # outputs = list(np.mean(output1.detach().cpu().numpy(),axis=(1,2,3)))
    print(ret/320)

showMask()
