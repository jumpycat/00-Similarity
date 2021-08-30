from face_utils import FaceDetector
import os
import torch
import cv2


target_root = r'I:\Celeb-DF/'
dst_path = r'I:\Celeb-DF_Images/'


face_detector = FaceDetector()
face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")
torch.set_grad_enabled(False)


def crop(video_path,src,file_name,IF_TRAIN):
    #'Face2Face','FaceSwap','NeuralTextures','Real'
    vid = cv2.VideoCapture(video_path)

    if IF_TRAIN:
        target_path = src.replace('Celeb-DF','Celeb-DF_Images') + '/' + 'train/'+ file_name[:-4]
    else:
        target_path = src.replace('Celeb-DF', 'Celeb-DF_Images') + '/' + 'test/' + file_name[:-4]
    if not os.path.exists(target_path):
        os.makedirs(target_path)


    gap = 5
    i = 0
    while True:
        success_df = vid.grab()
        if not success_df:
            break
        if i % gap == 0:
            df_success, frame = vid.retrieve()
            height, width, _ = frame.shape

            if frame is not None:
                boxes, landms = face_detector.detect(frame)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                l, t, r, b = boxes.tolist()
                h = b - t
                w = r - l
                maxl = int(max(h, w) * 1.6)
                centerx = (t + b) / 2
                centery = (l + r) / 2
                startx = centerx - maxl // 2
                starty = centery - maxl // 2



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

            df_face = frame[startx:startx + maxl, starty:starty + maxl, :]
            cv2.imwrite(target_path + '/' + str(i) + '.png', df_face)


        i += 1
    vid.release()

def cropall():
    test_list = r'I:\Celeb-DF\List_of_testing_videos.txt'
    test_viode_list = []
    with open(test_list, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            test_viode_list.append(line[2:])


    type = ['Celeb-real','Celeb-synthesis','YouTube-real']
    for j in type:
        src_ = target_root + j
        videos = os.listdir(src_)
        for ele in videos:
            video_path = src_ + '/' + ele
            pth = video_path.split('/')[1]+ '/' + video_path.split('/')[2]
            try:
                if pth in test_viode_list:
                    crop(video_path,src_,ele,False)
                else:
                    crop(video_path,src_,ele,True)
            except:
                print(ele)

cropall()

