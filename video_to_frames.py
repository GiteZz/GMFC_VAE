import numpy as np
import cv2
import scipy.misc
from PIL import Image


folder = "D:/Stage/Data/Avenue/testing_videos/"
file_name = "01.avi"

file = folder + file_name

cap = cv2.VideoCapture(file)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()


amount_frames = buf.shape[0]

for i in range(amount_frames):
    im = Image.fromarray(buf[i])
    im.save(folder + "01-Frames/" + str(i) + ".jpeg")
