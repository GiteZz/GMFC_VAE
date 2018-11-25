import numpy as np
from sklearn import svm

av_frames = 25

f_u = (1/av_frames) * np.load('u_buf.npy')
f_v = (1/av_frames) * np.load('v_buf.npy')


frame_amount = f_u.shape[0]
frame_height = f_u.shape[1]
frame_width = f_u.shape[2]
Fu = np.zeros((frame_amount - av_frames, frame_height, frame_width))
Fv = np.zeros((frame_amount - av_frames, frame_height, frame_width))
ft_u = np.zeros((frame_amount, frame_height, frame_width))
ft_v = np.zeros((frame_amount, frame_height, frame_width))

for i in range(av_frames):
    ft_u += np.roll(f_u, -i)
    ft_v += np.roll(f_u, -i)

slack_var = np.zeros((av_frames - 1, av_frames))
