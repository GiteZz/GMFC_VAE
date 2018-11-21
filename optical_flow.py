import numpy as np
from scipy import signal
from video_to_frames import video_to_numpy_grey_array

# code from
# https://stackoverflow.com/questions/14321092/lucas-kanade-python-numpy-implementation-uses-enormous-amount-of-memory
def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size / 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)


    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            # b = ... # get b here
            # A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ...  # get velocity here
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return (u, v)

if __name__ == "__main__":
    file = 'D:/Stage/Data/Avenue/testing_videos/01.avi'
    video_array = video_to_numpy_grey_array(file)
    test1 = optical_flow(video_array[0], video_array[1], 2)
    a = 5