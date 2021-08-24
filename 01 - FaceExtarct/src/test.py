import cv2

a = r'D:\DATA\FF++_Videos\Deepfakes\mask\masks\videos/001_870.mp4'

mask_video = cv2.VideoCapture(a)
mask_video.grab()


success_, mask_frame = mask_video.retrieve()
print(success_)