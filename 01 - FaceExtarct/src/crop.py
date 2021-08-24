from face_utils import FaceDetector
import os
import torch
import cv2
from config import *

face_detector = FaceDetector()
face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")
torch.set_grad_enabled(False)


def extract_frames(video_path, targetpath):
    raw_vid = cv2.VideoCapture(video_path)
    c23_vid = cv2.VideoCapture(video_path.replace('raw', 'c23'))
    c40_vid = cv2.VideoCapture(video_path.replace('raw', 'c40'))
    # msk_vid = cv2.VideoCapture(r'D:\DATA\FF++_Videos\Deepfakes\mask\masks\videos/' + video_path[-11:])
    gap = GAP
    scale = SCALE
    i = 0
    while True:
        success_raw = raw_vid.grab()
        success_c23 = c23_vid.grab()
        success_c40 = c40_vid.grab()
        # success_mask = msk_vid.grab()

        if not (success_raw and success_c23 and success_c40):
            break
        if i % gap == 0:
            raw_success, raw_frame = raw_vid.retrieve()
            c23_success, c23_frame = c23_vid.retrieve()
            c40_success, c40_frame = c40_vid.retrieve()
            # msk_success, msk_frame = msk_vid.retrieve()
            height, width, _ = raw_frame.shape
            if raw_frame is not None:
                boxes, landms = face_detector.detect(raw_frame)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                l, t, r, b = boxes.tolist()
                h = b - t
                w = r - l
                maxl = int(max(h, w) * scale)
                centerx = (t + b) / 2
                centery = (l + r) / 2
                startx = centerx - maxl // 2
                starty = centery - maxl // 2
                if startx <= 0:
                    startx = 0
                if startx + maxl >= height:
                    startx = height - maxl
                if starty <= 0:
                    starty = 0
                if starty + maxl >= width:
                    starty = width - maxl
                startx, starty = int(startx), int(starty)

                raw_face = raw_frame[startx:startx + maxl, starty:starty + maxl, :]
                c23_face = c23_frame[startx:startx + maxl, starty:starty + maxl, :]
                c40_face = c40_frame[startx:startx + maxl, starty:starty + maxl, :]
                # msk_face = msk_frame[startx:startx + maxl, starty:starty + maxl, :]

                if not os.path.exists(targetpath.replace('raw', 'c23')):
                    os.makedirs(targetpath.replace('raw', 'c23'))
                if not os.path.exists(targetpath.replace('raw', 'c40')):
                    os.makedirs(targetpath.replace('raw', 'c40'))

                cv2.imwrite(targetpath + '/' + str(i) + '.png', raw_face)
                cv2.imwrite(targetpath.replace('raw', 'c23') + '/' + str(i) + '.png', c23_face)
                cv2.imwrite(targetpath.replace('raw', 'c40') + '/' + str(i) + '.png', c40_face)

                # if not os.path.exists(r'D:\DATA\FF++Images\Deepfakes\mask/' + type + '/' + video_path[-11:-4]):
                #     os.makedirs(r'D:\DATA\FF++Images\Deepfakes\mask/' + type + '/' + video_path[-11:-4])
                # cv2.imwrite(r'D:\DATA\FF++Images\Deepfakes\mask/' + type + '/' + video_path[-11:-4] + '/' + str(i) + '.png', mask)
        i += 1
    raw_vid.release()
    c23_vid.release()
    c40_vid.release()


def cropfake():
    classes = ['c23', 'c40', 'raw']
    split = ['train', 'test', 'val']
    src = 'D:/DATA/FF++_Videos/Deepfakes/'

    dst_path = r'D:\DATA\FF++Images/'
    for i in classes:
        for j in split:
            src_ = src + i + '/' + j
            videos = os.listdir(src_)
            for m in videos:
                video_path = src_ + '/' + m

                file_name = m[:-4]
                dst_ = video_path[20:-4]
                dst_imgs_path = dst_path + dst_
                if not os.path.exists(dst_imgs_path):
                    os.makedirs(dst_imgs_path)
                try:
                    extract_frames(video_path, dst_imgs_path)
                except:
                    print(video_path, dst_imgs_path, j)


def cropreal():
    split = ['train', 'test', 'val']
    src = 'D:/DATA/FF++_Videos/Real/raw/'

    dst_path = r'D:\DATA\FF++_Images/'
    for j in split:
        src_ = src + j
        videos = os.listdir(src_)
        for m in videos:
            video_path = src_ + '/' + m
            file_name = m[:-4]
            dst_ = video_path[20:-4]  # Real\c23\test\001_780 类似

            dst_imgs_path = dst_path + dst_
            if not os.path.exists(dst_imgs_path):
                os.makedirs(dst_imgs_path)

            try:
                extract_frames(video_path, dst_imgs_path)
            except:
                print(video_path, dst_imgs_path)


cropreal()
