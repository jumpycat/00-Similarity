import cv2

a = r'D:\DATA\FF++_Videos/Face2Face/raw//train/019_018.mp4'

mask_video = cv2.VideoCapture(a)
mask_video.grab()


success_, mask_frame = mask_video.retrieve()
print(success_)