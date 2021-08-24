from face_utils import norm_crop, FaceDetector
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import sys

splitnum=int(sys.argv[1])
start=int(sys.argv[2])
print(os.getcwd())
face_detector = FaceDetector()

face_detector.load_checkpoint("../models/RetinaFace-Resnet50-fixed.pth")
torch.set_grad_enabled(False)

path='/Data/olddata_E/01DeepFakesDetection/03Celeb-DF_Full_Videos/'
target_path='/Data/olddata_E/01DeepFakesDetection/03Celeb-DF_Full_Videos/trainimg/'
def extract_frames(video_path,targetpath):
    """
    Extract frames from a video. You can use either provided method here or implement your own method.

    params:
        - video_local_path (str): the path of video.
    return:
        - frames (list): a list containing frames extracted from the video.
    """
    ########################################################################################################
    # You can change the lines below to implement your own frame extracting method (and possibly other preprocessing),
    # or just use the provided codes.
    import cv2

    vid = cv2.VideoCapture(video_path)

    cap=vid.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    detect_per_video=64
    scale=1.4
    i=0
    count=0
    while True:
        success = vid.grab()
        if not success:
            break
        if i%(cap//detect_per_video)==0:
            success,frame=vid.retrieve()
            height,width,_=frame.shape
            if frame is not None:
                boxes, landms = face_detector.detect(frame)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                l,t,r,b=boxes.tolist()
                h=b-t
                w=r-l
                maxl=int(max(h,w)*scale)
                centerx=(t+b)/2
                centery=(l+r)/2
                startx=centerx-maxl//2
                starty=centery-maxl//2
                if startx<=0:
                    startx=0
                if startx+maxl>=height:
                    startx=height-maxl
                if starty<=0:
                    starty=0
                if starty+maxl>=width:
                    starty=width-maxl
                startx,starty=int(startx),int(starty)
                face=frame[startx:startx+maxl,starty:starty+maxl,:]
                cv2.imwrite(targetpath+'/'+str(i)+'.png',face)
                count+=1
                if count>=detect_per_video:
                    break
                # img=cv2.resize(img,(224,224))
        i+=1
    vid.release()
    return frames
    ########################################################################################################
i=0
for dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
    dpath=path+dataset+'/'
    tpath=target_path+dataset+'/'
    for video in os.listdir(dpath):
        videopath=dpath+video
        target=tpath+video[:-4]
        if not os.path.exists(target):
            os.makedirs(target)
        try:
            extract_frames(videopath,target)
        except:
            print('error',target)
        print(i,videopath)
        i+=1
