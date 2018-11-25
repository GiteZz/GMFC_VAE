from keras.layers import Lambda, Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from get_mnist import get_number_mnist
from time import gmtime, strftime
import numpy as np

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_gmm_var_start(latent_dim, n_feat):
    # start with evenly distributed change for every Cat
    p_pi = K.variable(K.np.ones(n_feat) / n_feat)
    # values from gaussian
    mu = K.zeros(shape=(latent_dim, n_feat))
    sigma = K.ones(shape=(latent_dim, n_feat))

    return p_pi, mu, sigma

def calc_sigma(z):


# layer def
layer_sizes = [256, 256, 512]
latent_size = 10
n_feat = 10
input_dim = 784

x = Input(batch_shape=(input_dim,))
h = Dense(layer_sizes[0], activation='relu')(x)
h = Dense(layer_sizes[1], activation='relu')(h)
h = Dense(layer_sizes[2], activation='relu')(h)
z_mean = Dense(latent_size)(h)
z_log_var = Dense(latent_size)(h)
z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_log_var])
h_decoded = Dense(layer_sizes[-1], activation='relu')(z)
h_decoded = Dense(layer_sizes[-2], activation='relu')(h_decoded)
h_decoded = Dense(layer_sizes[-3], activation='relu')(h_decoded)
x_decoded_mean = Dense(input_dim, activation='sigmoid')(h_decoded)

p_pi, mu, sigma = get_gmm_var_start(latent_size, n_feat)