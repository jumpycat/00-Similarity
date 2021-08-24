import numpy as np
import cv2
import os

BATCH = 32
SIZE = 256


dst_train_path = r'D:\DATA\DeepFake\train/'
dst_test_path = r'D:\DATA\DeepFake\train/'

def train_gen():
    train_real_video_paths = os.listdir(dst_train_path+'real/')
    train_real_imgs = []
    for i in train_real_video_paths:
        video_path = dst_train_path+'real/'+i
        img = os.listdir(video_path)
        train_real_imgs.append([video_path+'/'+ j for j in img])

    train_fake_video_paths = os.listdir(dst_train_path+'fake/')
    train_fake_imgs = []
    for i in train_fake_video_paths:
        video_path = dst_train_path+'fake/'+i
        img = os.listdir(video_path)
        train_fake_imgs.append([video_path+'/'+ j for j in img])
    NUM_fake = len(train_fake_imgs)
    NUM_real = len(train_real_imgs)

    while True:
        data = []
        mask = []
        label = []
        for i in range(BATCH):
            if np.random.randint(0,2):
                video_index = np.random.randint(0,NUM_fake)
                img_index = np.random.randint(0,len(train_fake_imgs[video_index]))
                img_path = train_fake_imgs[video_index][img_index]
                data_img = cv2.resize(cv2.imread(img_path),(SIZE,SIZE))

                mask_video = img_path.split('/')[-2]
                mask_img = img_path.split('/')[-1]
                mask_path = r'D:\DATA\DeepFake\train\mask/'+str(mask_video)+'/'+str(mask_img)
                data_mask = np.expand_dims(np.array(cv2.resize(cv2.imread(mask_path,0),(SIZE,SIZE))>16,dtype=np.float64),axis=-1)

                data.append(data_img)
                mask.append(data_mask)
                label.append(1)
            else:
                video_index = np.random.randint(0,NUM_real)
                img_index = np.random.randint(0,len(train_real_imgs[video_index]))
                img_path = train_real_imgs[video_index][img_index]
                data_img = cv2.resize(cv2.imread(img_path),(SIZE,SIZE))
                data_mask = np.zeros((SIZE,SIZE,1))

                data.append(data_img)
                mask.append(data_mask)
                label.append(0)

            yield np.array(data),np.array(label)



