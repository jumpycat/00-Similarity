import os
import cv2
import numpy as np


target_root = r'I:\01-Dataset\01-Images\00-FF++\FaceShifter2.0/'

def crop(faces_vid_path):
    #'Face2Face','FaceSwap','NeuralTextures','Real'

    df_mask_path = r'I:\01-Dataset\00-Videos\00-FF++\Deepfakes\mask/'
    real_path = r'I:\01-Dataset\00-Videos\00-FF++\Real\raw/'


    df_mask_paths = [df_mask_path + m for m in os.listdir(df_mask_path)]



    real_paths = []
    splits = os.listdir(real_path)
    for m in splits:
        vids = os.listdir(real_path + m)
        real_paths += [real_path + m + '/' + el for el in vids]


    target_fs_face_path = '123'
    target_msk_face_path = '123'

    file_name = faces_vid_path[-11:-4] #001_780

    faces_vid = cv2.VideoCapture(faces_vid_path)
    for i in df_mask_paths:
        if file_name in i:
            msk_vid = cv2.VideoCapture(i)

    for j in real_paths:
        if file_name[:3] == j[-7:-4]:
            rel_vid = cv2.VideoCapture(j)
            if 'train' in j:
                target_msk_face_path = target_root + 'mask/train' + '/' + file_name
                target_fs_face_path = target_root + 'raw/train' + '/' + file_name
            elif 'test' in j:
                target_msk_face_path = target_root + 'mask/test' + '/' + file_name
                target_fs_face_path = target_root + 'raw/test' + '/' + file_name
            else:
                target_msk_face_path = target_root + 'mask/val' + '/' + file_name
                target_fs_face_path = target_root + 'raw/val' + '/' + file_name


    if not os.path.exists(target_fs_face_path):
        os.makedirs(target_fs_face_path)
    if not os.path.exists(target_msk_face_path):
        os.makedirs(target_msk_face_path)

    gap = 5
    scale = 3.4
    i = 0
    while True:
        success_df = faces_vid.grab()
        success_f2f = msk_vid.grab()
        success_fs = rel_vid.grab()


        if not (success_df and success_f2f and success_fs):
            break
        if i % gap == 0:
            faces_success, faces_frame = faces_vid.retrieve()
            smk_success, msk_frame = msk_vid.retrieve()
            rel_success, rel_frame = rel_vid.retrieve()

            df_msk_frame = cv2.cvtColor(msk_frame,cv2.COLOR_BGR2GRAY)>0

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


            rl_face = rel_frame[startx:startx + maxl, starty:starty + maxl, :]
            fs_face = faces_frame[startx:startx + maxl, starty:starty + maxl, :]
            sub = cv2.cvtColor(rl_face-fs_face, cv2.COLOR_BGR2GRAY) > 0

            cv2.imwrite(target_fs_face_path + '/' + str(i) + '.png', fs_face)
            cv2.imwrite(target_msk_face_path + '/' + str(i) + '.png', sub*255)


        i += 1
    faces_vid.release()
    msk_vid.release()
    rel_vid.release()

def cropall():
    split = ['train','val','test']


    faceshifter = r'I:\01-Dataset\00-Videos\00-FF++\FaceShifter\raw\videos/'

    faces_vid = os.listdir(faceshifter)
    for m in faces_vid:
        faces_vid_path = faceshifter + m
        try:
            crop(faces_vid_path)
        except:
            print(faces_vid_path)
cropall()
