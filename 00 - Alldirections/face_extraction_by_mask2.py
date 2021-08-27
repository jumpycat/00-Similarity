import os
import cv2
import numpy as np


target_root = r'D:\DATA\FF++_Images/'
dst_path = r'D:\DATA\FF++_Videos/'

def crop(video_path,type,split):
    file_name = video_path[-11:-4]

    raw_vid = cv2.VideoCapture(video_path)
    c23_vid = cv2.VideoCapture(video_path.replace('raw', 'c23'))
    c40_vid = cv2.VideoCapture(video_path.replace('raw', 'c40'))
    msk_vid = cv2.VideoCapture(dst_path + '/' + type + '/' + 'mask' + '/' + file_name + '.mp4')

    target_raw_face_path = target_root + '/' + type + '/' + 'raw' + '/' + split + '/' + file_name
    target_c23_face_path = target_root + '/' + type + '/' + 'c23' + '/' + split + '/' + file_name
    target_c40_face_path = target_root + '/' + type + '/' + 'c40' + '/' + split + '/' + file_name
    target_msk_face_path = target_root + '/' + type + '/' + 'mask' + '/' + split + '/' + file_name
    if not os.path.exists(target_raw_face_path):
        os.makedirs(target_raw_face_path)
    if not os.path.exists(target_c23_face_path):
        os.makedirs(target_c23_face_path)
    if not os.path.exists(target_c40_face_path):
        os.makedirs(target_c40_face_path)
    if not os.path.exists(target_msk_face_path):
        os.makedirs(target_msk_face_path)

    gap = 5
    scale = 1.8
    i = 0
    while True:
        success_raw = raw_vid.grab()
        success_c23 = c23_vid.grab()
        success_c40 = c40_vid.grab()
        success_msk = msk_vid.grab()

        if not (success_raw and success_c23 and success_c40 and success_msk):
            break
        if i % gap == 0:
            raw_success, raw_frame = raw_vid.retrieve()
            c23_success, c23_frame = c23_vid.retrieve()
            c40_success, c40_frame = c40_vid.retrieve()
            msk_success, msk_frame = msk_vid.retrieve()
            msk_frame = cv2.cvtColor(msk_frame,cv2.COLOR_BGR2GRAY)>0

            try:
                row = np.mean(msk_frame,axis=1)
                t = np.nonzero(row)[0][0]
                b = np.nonzero(row)[0][-1]

                colomn = np.mean(msk_frame,axis=0)
                l = np.nonzero(colomn)[0][0]
                r = np.nonzero(colomn)[0][-1]
            except:
                i += 1
                continue
            h = b - t
            w = r - l
            maxl = int(max(h, w) * scale)
            centerx = (t + b) / 2
            centery = (l + r) / 2
            startx = centerx - maxl // 2
            starty = centery - maxl // 2

            height, width, _ = raw_frame.shape
            if startx <= 0:
                startx = 0
            if startx + maxl >= height:
                startx = height - maxl
            if startx <= 0 and height - maxl <= 0:
                startx = 0
                maxl = height

            if starty <= 0:
                starty = 0
            if starty + maxl >= width:
                starty = width - maxl
            if starty <= 0 and width - maxl <= 0:
                starty = 0
                maxl = min(height,width)
            startx, starty = int(startx), int(starty)

            raw_face = raw_frame[startx:startx + maxl, starty:starty + maxl, :]
            c23_face = c23_frame[startx:startx + maxl, starty:starty + maxl, :]
            c40_face = c40_frame[startx:startx + maxl, starty:starty + maxl, :]
            msk_face = msk_frame[startx:startx + maxl, starty:starty + maxl]

            cv2.imwrite(target_raw_face_path + '/' + str(i) + '.png', raw_face)
            cv2.imwrite(target_c23_face_path + '/' + str(i) + '.png', c23_face)
            cv2.imwrite(target_c40_face_path + '/' + str(i) + '.png', c40_face)
            cv2.imwrite(target_msk_face_path + '/' + str(i) + '.png', np.array(msk_face,dtype=int)*255)
        i += 1
    raw_vid.release()
    c23_vid.release()
    c40_vid.release()
    msk_vid.release()

def cropreal():
    split = ['train', 'test', 'val']
    type = ['Face2Face','FaceSwap','NeuralTextures']
    for j in type:
        src_ = dst_path + j + r'\\' + 'raw' + '/'
        for k in split:
            src__ = src_ + '/' + k
            videos = os.listdir(src__)
            for m in videos:
                video_path = src__ + r'\\' + m
                try:
                    crop(video_path, j,k)
                except:
                    print(video_path)
# cropreal()

dic = [
'D:\DATA\FF++_Videos/Face2Face\\raw//train\\629_618.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\096_101.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\101_096.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\155_576.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\317_359.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\359_317.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\449_451.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\451_449.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\509_525.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\525_509.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\547_574.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\574_547.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\576_155.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\601_653.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\618_629.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\629_618.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\653_601.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\665_679.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\679_665.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\704_723.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\723_704.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\738_804.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\764_850.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\804_738.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//train\\850_764.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\135_880.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\170_186.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\186_170.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\462_467.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\467_462.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//test\\880_135.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//val\\370_483.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//val\\483_370.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//val\\672_720.mp4',
# 'D:\DATA\FF++_Videos/FaceSwap\\raw//val\\720_672.mp4'
]

for ele in dic:
    crop(ele, 'Face2Face', 'train')