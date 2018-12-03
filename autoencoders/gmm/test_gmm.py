from keras.layers import Lambda, Input, Dense, Conv2D, Flatten
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from get_mnist import get_number_mnist, get_small_test_batch
from time import gmtime, strftime
import numpy as np

total = np.load('total.npy')
feat = np.load('feat.npy')
a = 5