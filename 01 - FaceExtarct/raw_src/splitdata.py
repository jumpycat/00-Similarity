import os
import shutil

spath1='/home/amax/ypp/deepfake/compress_videos_imgs/Real/raw/videos/'
trainlist1=[s for s in os.listdir(spath1+'train/')]
vallist1=[s for s in os.listdir(spath1+'val/')]
testlist1=[s for s in os.listdir(spath1+'test/')]

datapath='/Data/olddata_E/01DeepFakesDetection/04FF++/compress_videos/'
for dataset in os.listdir(datapath):
    datasetpath=datapath+dataset
    if dataset=='Real':
        for c in ['c23','c40']:
            i=0
            cpath=datasetpath+'/'+c+'/videos/'
            for video in os.listdir(cpath):
                videopath=cpath+video
                namesplit=video.split('.')
                if len(namesplit)==2:
                    v1,v2=namesplit
                    if v1 in trainlist1:
                        targetpath=datasetpath+'/'+c+'/train/'
                        i+=1
                    elif v1 in vallist1:
                        targetpath=datasetpath+'/'+c+'/val/'
                    elif v1 in testlist1:
                        targetpath=datasetpath+'/'+c+'/test/'
                    if not os.path.exists(targetpath):
                        os.makedirs(targetpath)
                    shutil.move(videopath,targetpath+video)
                # print(targetpath)
            print(i)

