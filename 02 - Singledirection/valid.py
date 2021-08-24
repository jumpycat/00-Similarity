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
import matplotlib.pyplot as plt




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ret = np.ones((m - 1, n))
    for i in range(m - 1):
        for j in range(n):
            up = 1 - abs(x[i, j] - x[i + 1, j])
            ret[i, j] = up
    return ret


def getValdata(size):
    dst_test_path = r'D:\DATA\DeepFake\train/'
    test_real_video_paths = os.listdir(dst_test_path + 'real/')
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = dst_test_path + 'real/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    test_fake_video_paths = os.listdir(dst_test_path + 'fake/')
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = dst_test_path + 'fake/' + i
        img = os.listdir(video_path)
        test_fake_imgs.append([video_path + '/' + j for j in img])
    NUM_fake = len(test_fake_imgs)
    NUM_real = len(test_real_imgs)

    imgs = []
    for i in range(size):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, NUM_fake)
            img_index = np.random.randint(0, len(test_fake_imgs[video_index]))
            img_path = test_fake_imgs[video_index][img_index]
            img = default_loader(img_path)
            imgs.append(img)
            # mask_video = img_path.split('/')[-2]
            # mask_img = img_path.split('/')[-1]
            # mask_path = r'D:\DATA\DeepFake\test\mask/' + str(mask_video) + '/' + str(mask_img)
            #
            # fake_mask = cv2.imread(mask_path, 0)
            # fake_mask = np.array(cv2.resize(fake_mask, (SIZE, SIZE)) > 1, dtype=np.float64)
            # fake_mask1 = pool2d(fake_mask, 16, 16)
            # fake_mask = Calsim(fake_mask1)
            # fake_mask = np.expand_dims(fake_mask, 0)
            #
            # mask = torch.from_numpy(fake_mask)
            # mask = torch.tensor(mask, dtype=torch.float32)
            # label.append(1)
        else:
            video_index = np.random.randint(0, NUM_real)
            img_index = np.random.randint(0, len(test_real_imgs[video_index]))
            img_path = test_real_imgs[video_index][img_index]
            img = default_loader(img_path)
            imgs.append(img)
            # mask = torch.ones((1, 15, 16), dtype=torch.float32)
            # label.append(0)
    return torch.stack(imgs, dim=0)


# print(getValdata(100).shape)

net = torch.load('model.pkl')

net.to(device)

ret = []
for i in range(20):
    inputs = getValdata(64)
    outputs = inputs.to(device)
    outputs = net(outputs)

    # print(np.around(outputs[0][0].detach().cpu().numpy(),2))
    outputs = list(np.mean(net(outputs).detach().cpu().numpy(),axis=(1,2,3)))
    ret += outputs
print(ret)
#
plt.hist(ret, bins=50)
plt.xlabel('mean')
plt.ylabel('num')
plt.show()