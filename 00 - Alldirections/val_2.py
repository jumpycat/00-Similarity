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

# Deepfakes Face2Face FaceSwap NeuralTextures
Fake_root = r'H:\FF++_Images_v2\FaceSwap\raw\val'
net = torch.load(r'trained_models\fs_v2\epoch-019-loss-0.091.pkl')

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
    real_root = r'H:\FF++_Images_v2\Real\raw\val'
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
        inputs,labels = getValdata(50)
        input = inputs.cuda()
        output1,output2 = net(input)
        outputs = np.array(np.mean(torch.sigmoid(output1.detach().cpu()).numpy(),axis=(1,2,3))>0.9,dtype=int)
        res = np.sum(outputs==np.array(labels))
        ret += res
        # outputs = list(np.mean(output1.detach().cpu().numpy(),axis=(1,2,3)))
    print(ret/320)

def findthrehold(pred,label):
    best_acc = 0
    for th in [0.8 + mom/1000 for mom in range(200)]:
        threhold_acc = np.array(np.array(pred)>th,dtype=int)
        acc = np.sum(threhold_acc == np.array(label))/3200
        if acc > best_acc:
            best_acc = acc
            print(th,best_acc)

def showHISTandMsk():
    ret_hist = []
    ret_labels = []
    for i in range(100):
        inputs, label = getValdata(32)
        input = inputs.cuda()
        output1, output2 = net(input)

        up = torch.sigmoid(output1).detach().cpu().numpy()[:,:,:-1,1:15]
        down = torch.sigmoid(output1).detach().cpu().numpy()[:,:,1:,1:15]
        left = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:15,:-1]
        right = torch.sigmoid(output2).detach().cpu().numpy()[:,:,1:15,1:]

        sim_map = np.mean(np.concatenate((up, down, left, right), axis=1),axis=(1,2,3))
        batch_sim_map_avg = list(sim_map)

        ret_hist += batch_sim_map_avg
        ret_labels += label

    findthrehold(ret_hist, ret_labels)

    threhold_acc = np.array(np.array(ret_hist) > 0.907, dtype=int)
    acc = np.sum(threhold_acc == np.array(ret_labels)) / 3200
    print('not val:',acc)

    # 0.995 / 0.967
    # 0.9131 / 0.966   0.9119
    plt.hist(ret_hist, bins=100)
    plt.xlabel('mean')
    plt.ylabel('num')
    plt.show()


if __name__ == '__main__':
    showHISTandMsk()
