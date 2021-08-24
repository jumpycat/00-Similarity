from face_utils import FaceDetector
import os
import torch
import cv2
import sys



splitnum=int(sys.argv[1])
start=int(sys.argv[2])
print(os.getcwd())


face_detector = FaceDetector()

face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")
torch.set_grad_enabled(False)

def extract_frames(video_path,mask_path,targetpath,target_maskpath):
    video_path_c23=video_path.replace('raw','c23')
    video_path_c40=video_path.replace('raw','c40')
    targetpath_c23=target_path.replace('raw','c23')
    targetpath_c40=target_path.replace('raw','c40')

    if not os.path.exists(targetpath_c23):
        os.makedirs(targetpath_c23)

    if not os.path.exists(targetpath_c40):
        os.makedirs(targetpath_c40)

    vid = cv2.VideoCapture(video_path)
    vid_c23= cv2.VideoCapture(video_path_c23)
    vid_c40= cv2.VideoCapture(video_path_c40)
    vid_mask = cv2.VideoCapture(mask_path)

    cap=vid.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    detect_per_video=32
    scale=1.4
    i=0
    count=0
    while True:
        success = vid.grab()
        success_c23 = vid_c23.grab()
        success_c40 = vid_c40.grab()
        success_mask = vid_mask.grab()
        if not success:
            break
        if i%(cap//detect_per_video)==0:
            success,frame=vid.retrieve()
            success_c23,frame_c23=vid_c23.retrieve()
            success_c40,frame_c40=vid_c40.retrieve()
            success_mask,frame_mask=vid_mask.retrieve()
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
                face_c23=frame_c23[startx:startx+maxl,starty:starty+maxl,:]
                face_c40=frame_c40[startx:startx+maxl,starty:starty+maxl,:]
                mask=frame_mask[startx:startx+maxl,starty:starty+maxl,:]
                cv2.imwrite(targetpath+'/'+str(i)+'.png',face)
                cv2.imwrite(targetpath_c23+'/'+str(i)+'.png',face_c23)
                cv2.imwrite(targetpath_c40+'/'+str(i)+'.png',face_c40)
                cv2.imwrite(target_maskpath+'/'+str(i)+'.png',mask)
                count+=1
                if count>=detect_per_video:
                    break
                # img=cv2.resize(img,(224,224))
        i+=1
    vid.release()
    vid_c23.release()
    vid_c40.release()
    vid_mask.release()
    return frames
    ########################################################################################################


# /Data/olddata_D/ypp/mask/
# path='/Data/olddata_E/01DeepFakesDetection/04FF++/compress_videos/Deepfakes/'
# extract_frames(path+'c23/videos/350_349.mp4',path+'mask/masks/videos/350_349.mp4')

datapath='/Data/olddata_E/01DeepFakesDetection/04FF++/compress_videos/'
targetpath='/Data/olddata_D/ypp/mask/ff++/'
processdata=[]
i=0
for dataset in os.listdir(datapath):
    if dataset!='Deepfakes' and dataset !='mask':
        datasetpath=datapath+dataset
        for c in ['raw']:
            cpath=datasetpath+'/'+c
            for s in os.listdir(cpath):
                spath=cpath+'/'+s
                for video in os.listdir(spath):
                    if '.' in video:
                        videopath=spath+'/'+video
                        maskpath=datasetpath+'/mask/masks/videos/'+video
                        target_path=targetpath+dataset+'/'+c+'/'+s+'/'+video[:-4]
                        target_maskpath=targetpath+dataset+'/mask/'+video[:-4]
                        processdata.append([videopath,maskpath,target_path,target_maskpath])
                        i+=1
print('total:',i)
processdata.sort()
processdatalen=len(processdata)
d=processdatalen//splitnum
if start!=splitnum-1:
    data=processdata[start*d:(start+1)*d]
else:
    data=processdata[start*d:]
i=0
datalen=len(data)
for video in data:
    videopath=video[0]
    maskpath=video[1]
    target_path=video[2]
    target_maskpath=video[3]
    video_name=videopath.split('/')[-1]
    video_quality=videopath.split('/')[-2]
    video_split=videopath.split('/')[-3]
    video_dataset=videopath.split('/')[-4]

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if len(os.listdir(target_path))>=32:
        continue
    if not os.path.exists(target_maskpath):
        os.makedirs(target_maskpath)
    try:
        extract_frames(videopath,maskpath,target_path,target_maskpath)
    except:
        print('error',target_path)
    i+=1
    print(i,'/',datalen,video_dataset,video_quality,video_split,video_name)
