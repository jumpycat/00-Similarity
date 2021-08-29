import os
import cv2
import numpy as np


target_root = r'H:\FF++_Images_v2/'
dst_path = r'D:\DATA\FF++_Videos/'

def crop(video_path,type,split):
    #'Face2Face','FaceSwap','NeuralTextures','Real'

    file_name = video_path[-11:-4]
    df_vid = cv2.VideoCapture(video_path)
    f2f_vid = cv2.VideoCapture(video_path.replace('Deepfakes', 'Face2Face'))
    fs_vid = cv2.VideoCapture(video_path.replace('Deepfakes', 'FaceSwap'))
    nt_vid = cv2.VideoCapture(video_path.replace('Deepfakes', 'NeuralTextures'))
    rel_vid = cv2.VideoCapture(video_path.replace('Deepfakes', 'Real')[:-11] + '/' + file_name[:3] + '.mp4')

    df_msk_vid = cv2.VideoCapture(dst_path + '/' + type + '/' + 'mask' + '/' + file_name + '.mp4')
    f2f_msk_vid = cv2.VideoCapture(dst_path + '/' + 'Face2Face' + '/' + 'mask' + '/' + file_name + '.mp4')
    fs_msk_vid = cv2.VideoCapture(dst_path + '/' + 'FaceSwap' + '/' + 'mask' + '/' + file_name + '.mp4')
    nt_msk_vid = cv2.VideoCapture(dst_path + '/' + 'NeuralTextures' + '/' + 'mask' + '/' + file_name + '.mp4')

    target_df_face_path = target_root + '/' + type + '/' + 'raw' + '/' + split + '/' + file_name
    target_f2f_face_path = target_root + '/' + 'Face2Face' + '/' + 'raw' + '/' + split + '/' + file_name
    target_fs_face_path = target_root + '/' + 'FaceSwap' + '/' + 'raw' + '/' + split + '/' + file_name
    target_nt_face_path = target_root + '/' + 'NeuralTextures' + '/' + 'raw' + '/' + split + '/' + file_name
    target_rel_face_path = target_root + '/' + 'Real' + '/' + 'raw' + '/' + split + '/' + file_name[:3]

    target_df_msk_path = target_root + '/' + type + '/' + 'mask' + '/' + split + '/' + file_name
    target_f2f_msk_path = target_root + '/' + 'Face2Face' + '/' + 'mask' + '/' + split + '/' + file_name
    target_fs_msk_path = target_root + '/' + 'FaceSwap' + '/' + 'mask' + '/' + split + '/' + file_name
    target_nt_msk_path = target_root + '/' + 'NeuralTextures' + '/' + 'mask' + '/' + split + '/' + file_name


    if not os.path.exists(target_df_face_path):
        os.makedirs(target_df_face_path)
    if not os.path.exists(target_f2f_face_path):
        os.makedirs(target_f2f_face_path)
    if not os.path.exists(target_fs_face_path):
        os.makedirs(target_fs_face_path)
    if not os.path.exists(target_nt_face_path):
        os.makedirs(target_nt_face_path)
    if not os.path.exists(target_rel_face_path):
        os.makedirs(target_rel_face_path)

    if not os.path.exists(target_df_msk_path):
        os.makedirs(target_df_msk_path)
    if not os.path.exists(target_f2f_msk_path):
        os.makedirs(target_f2f_msk_path)
    if not os.path.exists(target_fs_msk_path):
        os.makedirs(target_fs_msk_path)
    if not os.path.exists(target_nt_msk_path):
        os.makedirs(target_nt_msk_path)

    gap = 5
    scale = 2.6
    i = 0
    while True:
        success_df = df_vid.grab()
        success_f2f = f2f_vid.grab()
        success_fs = fs_vid.grab()
        success_nt = nt_vid.grab()
        success_rel = rel_vid.grab()

        success_df_msk = df_msk_vid.grab()
        success_f2f_msk = f2f_msk_vid.grab()
        success_fs_msk = fs_msk_vid.grab()
        success_nt_msk = nt_msk_vid.grab()


        if not (success_df and success_f2f and success_fs and success_nt and success_rel
                and success_df_msk and success_f2f_msk and success_fs_msk and success_nt_msk):
            break
        if i % gap == 0:
            df_success, df_frame = df_vid.retrieve()
            f2f_success, f2f_frame = f2f_vid.retrieve()
            fs_success, fs_frame = fs_vid.retrieve()
            nt_success, nt_frame = nt_vid.retrieve()
            rel_success, rel_frame = rel_vid.retrieve()

            df_msk_success, df_msk_frame = df_msk_vid.retrieve()
            f2f_msk_success, f2f_msk_frame = f2f_msk_vid.retrieve()
            fs_msk_success, fs_msk_frame = fs_msk_vid.retrieve()
            nt_msk_success, nt_msk_frame = nt_msk_vid.retrieve()


            df_msk_frame = cv2.cvtColor(df_msk_frame,cv2.COLOR_BGR2GRAY)>0
            f2f_msk_frame = cv2.cvtColor(f2f_msk_frame,cv2.COLOR_BGR2GRAY)>0
            fs_msk_frame = cv2.cvtColor(fs_msk_frame,cv2.COLOR_BGR2GRAY)>0
            nt_msk_frame = cv2.cvtColor(nt_msk_frame,cv2.COLOR_BGR2GRAY)>0

            try:
                row = np.mean(df_msk_frame,axis=1)
                t = np.nonzero(row)[0][0]
                b = np.nonzero(row)[0][-1]

                colomn = np.mean(df_msk_frame,axis=0)
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

            height, width = df_msk_frame.shape
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

            df_face = df_frame[startx:startx + maxl, starty:starty + maxl, :]
            f2f_face = f2f_frame[startx:startx + maxl, starty:starty + maxl, :]
            fs_face = fs_frame[startx:startx + maxl, starty:starty + maxl, :]
            nt_face = nt_frame[startx:startx + maxl, starty:starty + maxl, :]
            rel_face = rel_frame[startx:startx + maxl, starty:starty + maxl, :]

            df_msk = df_msk_frame[startx:startx + maxl, starty:starty + maxl]
            f2f_msk = f2f_msk_frame[startx:startx + maxl, starty:starty + maxl]
            fs_msk = fs_msk_frame[startx:startx + maxl, starty:starty + maxl]
            nt_msk = nt_msk_frame[startx:startx + maxl, starty:starty + maxl]



            cv2.imwrite(target_df_face_path + '/' + str(i) + '.png', df_face)
            cv2.imwrite(target_f2f_face_path + '/' + str(i) + '.png', f2f_face)
            cv2.imwrite(target_fs_face_path + '/' + str(i) + '.png', fs_face)
            cv2.imwrite(target_nt_face_path + '/' + str(i) + '.png', nt_face)
            cv2.imwrite(target_rel_face_path + '/' + str(i) + '.png', rel_face)
            cv2.imwrite(target_df_msk_path + '/' + str(i) + '.png', np.array(df_msk,dtype=int)*255)
            cv2.imwrite(target_f2f_msk_path + '/' + str(i) + '.png', np.array(f2f_msk,dtype=int)*255)
            cv2.imwrite(target_fs_msk_path + '/' + str(i) + '.png', np.array(fs_msk,dtype=int)*255)
            cv2.imwrite(target_nt_msk_path + '/' + str(i) + '.png', np.array(nt_msk,dtype=int)*255)

        i += 1
    df_vid.release()
    f2f_vid.release()
    fs_vid.release()
    nt_vid.release()
    rel_vid.release()

    df_msk_vid.release()
    f2f_msk_vid.release()
    fs_msk_vid.release()
    nt_msk_vid.release()


def cropall():
    split = ['train','val']
    type = ['Deepfakes']
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
cropall()
