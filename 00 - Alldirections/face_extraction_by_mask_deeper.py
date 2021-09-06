import os
import cv2
import numpy as np


target_root = r'I:\01-Dataset\01-Images\00-FF++/'
dst_path = r'I:\01-Dataset\00-Videos\00-FF++/'
deeper_root = r'I:\01-Dataset\00-Videos\04-DeeperForensics-1.0\manipulated_videos\end_to_end/'
deerper_paths = [deeper_root + i for i in os.listdir(deeper_root)]


def crop(video_path,type,split):
    #'Face2Face','FaceSwap','NeuralTextures','Real'

    file_name = video_path[-11:-4]
    # rel_vid = cv2.VideoCapture(video_path.replace('Deepfakes', 'Real')[:-11] + '/' + file_name[:3] + '.mp4')
    df_msk_vid = cv2.VideoCapture(dst_path + '/' + type + '/' + 'mask' + '/' + file_name + '.mp4')

    for pp in deerper_paths:
        if file_name[:3] == pp[-12:-9]:
            deerper_path = pp
    deeper_vid = cv2.VideoCapture(deerper_path)

    target_df_face_path = r'D:\DATA\Deeper/' + 'raw' + '/' + split + '/' + file_name
    # target_msk_path = r'D:\DATA\Deeper/' + 'mask' + '/' + split + '/' + file_name


    if not os.path.exists(target_df_face_path):
        os.makedirs(target_df_face_path)
    # if not os.path.exists(target_msk_path):
    #     os.makedirs(target_msk_path)

    gap = 5
    scale = 2.6
    i = 0
    while True:
        # success_rel = rel_vid.grab()
        success_df_msk = df_msk_vid.grab()
        success_deeper = deeper_vid.grab()

        if not (success_df_msk and success_deeper):
            break
        if i % gap == 0:
            # df_success, rel_frame = rel_vid.retrieve()
            f2f_success, df_msk_frame = df_msk_vid.retrieve()
            deeper_success,  deeper_frame = deeper_vid.retrieve()

            df_msk_frame = cv2.cvtColor(df_msk_frame,cv2.COLOR_BGR2GRAY)>0

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



            deepr_face = deeper_frame[startx:startx + maxl, starty:starty + maxl, :]
            # rl_face = rel_frame[startx:startx + maxl, starty:starty + maxl, :]

            # sub = cv2.cvtColor(rl_face-deepr_face, cv2.COLOR_BGR2GRAY) > 20


            cv2.imwrite(target_df_face_path + '/' + str(i) + '.png', deepr_face)
            # cv2.imwrite(target_msk_path + '/' + str(i) + '.png', sub*255)


        i += 1
    # rel_vid.release()
    deeper_vid.release()
    df_msk_vid.release()



def cropall():
    split = ['train','val','test']
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
