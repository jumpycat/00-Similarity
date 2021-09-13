import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import torch
from torch.utils.data import Dataset
import cv2
import random

SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor()
])


def val_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


def default_loader(path, new_startx, new_starty, HW):
    img_pil = Image.open(path)
    if np.random.randint(1, 4) % 3 == 0:
        if np.random.randint(1, 7) == 1:
            least = np.random.randint(48, 160)
            img_pil = img_pil.resize((least, least), Image.ANTIALIAS)
        if np.random.randint(1, 7) == 1:
            least = np.random.randint(48, 160)
            img_pil = img_pil.resize((least, least), Image.NEAREST)
        if np.random.randint(1, 7) == 1:
            least = np.random.randint(48, 160)
            img_pil = img_pil.resize((least, least), Image.BILINEAR)
        if np.random.randint(1, 7) == 1:
            least = np.random.randint(48, 160)
            img_pil = img_pil.resize((least, least), Image.BICUBIC)

    img_pil = img_pil.crop([new_startx, new_starty, new_startx + HW, new_starty + HW])
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
    ret = np.ones((m - 1, n))
    for i in range(m - 1):
        for j in range(n):
            if abs(x[i + 1, j] - x[i, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimleft(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n - 1))
    for i in range(m):
        for j in range(n - 1):
            if abs(x[i, j + 1] - x[i, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimup_bank(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n + 2))
    for i in range(2, m):
        for j in range(1, n - 1):
            if abs(x[i, j] - x[i - 2, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimleft_bank(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m + 2, n))
    for i in range(1, m - 1):
        for j in range(2, n):
            if abs(x[i, j] - x[i, j - 2]) > 0:
                ret[i, j] = 0
    return ret


class DealDataset(Dataset):
    def __init__(self, TRAIN_FAKE_ROOT, TRAIN_REAL_ROOT, LENGTH, TYPE, loader=default_loader):
        self.len = LENGTH
        self.loader = loader
        self.fake_root = TRAIN_FAKE_ROOT
        self.real_root = TRAIN_REAL_ROOT
        self.TYPE = TYPE

        train_fake_video_paths = os.listdir(self.fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = self.fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        train_real_video_paths = os.listdir(self.real_root)
        self.train_real_imgs = []
        for i in train_real_video_paths:
            video_path = self.real_root + i
            img = os.listdir(video_path)
            self.train_real_imgs.append([video_path + '/' + j for j in img])
        self.NUM_fake = len(self.train_fake_imgs)
        self.NUM_real = len(self.train_real_imgs)

    def __getitem__(self, index):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, self.NUM_fake)
            img_index = np.random.randint(0, len(self.train_fake_imgs[video_index]))
            img_path = self.train_fake_imgs[video_index][img_index]

            mask_path = img_path.replace(self.TYPE, 'mask')

            fake_mask = cv2.imread(mask_path, 0)
            _ = fake_mask.shape[0]
            if np.random.randint(1, 4) % 3 == 0:
                new_startx = random.randint(0, int(0.2 * _))
                new_starty = random.randint(0, int(0.2 * _))
                HW = random.randint(int(0.7 * _), _ - max(new_startx, new_starty) - 1)
            else:
                new_startx, new_starty, HW = 0, 0, _

            img = self.loader(img_path, new_startx, new_starty, HW)

            fake_mask = fake_mask[new_startx:new_startx + HW, new_starty:new_starty + HW]
            fake_mask = np.array(cv2.resize(fake_mask, (SIZE, SIZE)) > 1, dtype=np.float64)
            fake_mask1 = pool2d(fake_mask, 16, 16)

            fake_mask_up = Calsimup(fake_mask1)
            fake_mask_left = Calsimleft(fake_mask1)
            fake_mask_up = torch.from_numpy(np.expand_dims(fake_mask_up, 0))
            fake_mask_left = torch.from_numpy(np.expand_dims(fake_mask_left, 0))
            # fake_mask_up = fake_mask_up.repeat(64,1,1)
            # fake_mask_left = fake_mask_left.repeat(64,1,1)
            mask_up = torch.tensor(fake_mask_up, dtype=torch.float32)
            mask_left = torch.tensor(fake_mask_left, dtype=torch.float32)

            fake_mask_up_bank = Calsimup_bank(fake_mask1)
            fake_mask_left_bank = Calsimleft_bank(fake_mask1)
            fake_mask_up_bank = torch.from_numpy(np.expand_dims(fake_mask_up_bank, 0))
            fake_mask_left_bank = torch.from_numpy(np.expand_dims(fake_mask_left_bank, 0))
            # fake_mask_up_bank = fake_mask_up_bank.repeat(64,1,1)
            # fake_mask_left_bank = fake_mask_left_bank.repeat(64,1,1)
            mask_up_bank = torch.tensor(fake_mask_up_bank, dtype=torch.float32)
            mask_left_bank = torch.tensor(fake_mask_left_bank, dtype=torch.float32)

            #
            # fake_mask_up2 = Calsimup(mask_up[0])
            # fake_mask_left2 = Calsimleft(mask_left[0])
            # fake_mask_up2 = torch.from_numpy(np.expand_dims(fake_mask_up2, 0))
            # fake_mask_left2 = torch.from_numpy(np.expand_dims(fake_mask_left2, 0))
            # mask_up2 = torch.tensor(fake_mask_up2, dtype=torch.float32)
            # mask_left2 = torch.tensor(fake_mask_left2, dtype=torch.float32)
            #
            # fake_mask_up_bank2 = Calsimup_bank(mask_up_bank[0])
            # fake_mask_left_bank2 = Calsimleft_bank(mask_left_bank[0])
            # fake_mask_up_bank2 = torch.from_numpy(np.expand_dims(fake_mask_up_bank2, 0))
            # fake_mask_left_bank2 = torch.from_numpy(np.expand_dims(fake_mask_left_bank2, 0))
            # mask_up_bank2 = torch.tensor(fake_mask_up_bank2, dtype=torch.float32)
            # mask_left_bank2 = torch.tensor(fake_mask_left_bank2, dtype=torch.float32)

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            _ = cv2.imread(img_path, 0).shape[0]

            if np.random.randint(1, 4) % 3 == 0:
                new_startx = random.randint(0, int(0.2 * _))
                new_starty = random.randint(0, int(0.2 * _))
                HW = random.randint(int(0.7 * _), _ - max(new_startx, new_starty) - 1)
            else:
                new_startx, new_starty, HW = 0, 0, _

            img = self.loader(img_path, new_startx, new_starty, HW)

            mask_up = torch.ones((1, 15, 16), dtype=torch.float32)
            mask_left = torch.ones((1, 16, 15), dtype=torch.float32)
            mask_up_bank = torch.ones((1, 16, 18), dtype=torch.float32)
            mask_left_bank = torch.ones((1, 18, 16), dtype=torch.float32)

            # mask_up2 = torch.ones((1, 15, 16), dtype=torch.float32)
            # mask_left2 = torch.ones((1, 16, 15), dtype=torch.float32)
            # mask_up_bank2 = torch.ones((1, 16, 18), dtype=torch.float32)
            # mask_left_bank2 = torch.ones((1, 18, 16), dtype=torch.float32)

        return img, (mask_up, mask_up_bank, mask_left, mask_left_bank)
        # ,mask_up2, mask_up_bank2, mask_left2, mask_left_bank2)

    def __len__(self):
        return self.len


def getDataset(VAL_REAL_ROOT, VAL_FAKE_ROOT):
    real_root = VAL_REAL_ROOT
    test_real_video_paths = os.listdir(real_root)
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = real_root + '/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    fake_root = VAL_FAKE_ROOT
    test_fake_video_paths = os.listdir(fake_root)
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = fake_root + '/' + i
        img = os.listdir(video_path)
        test_fake_imgs.append([video_path + '/' + j for j in img])

    NUM_fake = len(test_fake_imgs)
    NUM_real = len(test_real_imgs)
    return NUM_fake, NUM_real, test_fake_imgs, test_real_imgs


def getValdata(size, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs):
    imgs = []
    labels = []
    for i in range(size):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, NUM_fake)
            img_index = np.random.randint(0, len(test_fake_imgs[video_index]))
            img_path = test_fake_imgs[video_index][img_index]
            img = val_loader(img_path)
            imgs.append(img)
            labels.append(0)
        else:
            video_index = np.random.randint(0, NUM_real)
            img_index = np.random.randint(0, len(test_real_imgs[video_index]))
            img_path = test_real_imgs[video_index][img_index]
            img = val_loader(img_path)
            imgs.append(img)
            labels.append(1)

    return torch.stack(imgs, dim=0), labels


def findthrehold(pred, label):
    best_acc = 0
    best_th = 0
    for th in [0.8 + mom / 1000 for mom in range(200)]:
        threhold_acc = np.array(np.array(pred) > th, dtype=int)
        acc = np.sum(threhold_acc == np.array(label)) / 1000
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def Val(model, VAL_REAL_ROOT, VAL_FAKE_ROOT):
    model.eval()
    NUM_fake, NUM_real, test_fake_imgs, test_real_imgs = getDataset(VAL_REAL_ROOT, VAL_FAKE_ROOT)

    ret_hist = []
    ret_labels = []
    for i in range(20):
        inputs, label = getValdata(50, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs)
        input = inputs.cuda()
        output1, output2, output3, output4 = model(input)

        up = torch.sigmoid(output1).detach().cpu().numpy()[:, :, :-1, 1:-1]
        down = torch.sigmoid(output1).detach().cpu().numpy()[:, :, 1:, 1:-1]
        left = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, :-1]
        right = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, 1:]

        up_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
        down_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
        left_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]
        right_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]

        sim_map = np.mean(np.concatenate((up, down, left, right, up_bank, down_bank, left_bank, right_bank), axis=1),
                          axis=(1, 2, 3))
        batch_sim_map_avg = list(sim_map)

        ret_hist += batch_sim_map_avg
        ret_labels += label

    auc = calcAUC_byProb(ret_labels, ret_hist)
    acc, th = findthrehold(ret_hist, ret_labels)
    print('Threshold:', np.round(th, 3), 'Accuracy:', np.round(acc * 100, 2), 'AUC:', np.round(auc, 4))
    return acc, th,auc


def calcAUC_byProb(labels, probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []
    for _, i in enumerate(labels):
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
