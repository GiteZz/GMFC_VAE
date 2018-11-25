from keras.layers import Lambda, Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from get_mnist import get_number_mnist, get_small_test_batch
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
import math
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

# (15) and (16) in paper
def calc_gamma(z_latent):
    """
    First we should transform everything to shape=(batch_size, latent_dim, n_feat)
    then we use the eq (15) and (16) from the paper

    z is the latent representation from the input x

    return the gamma with shape=(batch_size, n_feat)
    """
    print(mu.get_shape())
    print(batch_size)
    print("end")


    z_3 = tf.transpose(K.repeat(z_latent, n_feat), [0, 2, 1])
    #mu_3 = tf.reshape(tf.tile(mu, batch_size), (batch_size, latent_size, n_feat))
    mu_3 = K.repeat_elements(tf.expand_dims(mu, 0), batch_size, axis=0)
    sigma_3 = K.repeat_elements(tf.expand_dims(sigma, 0), batch_size, axis=0)
    p_pi_3 = K.repeat_elements(tf.expand_dims(p_pi, 0), latent_size, axis=0)
    p_pi_3 = K.repeat_elements(tf.expand_dims(p_pi_3, 0), batch_size, axis=0)

    inner_part = K.log(p_pi_3) - 0.5 * K.log(2 * math.pi * sigma_3) - K.square(z_3 - mu_3) / (2 * sigma_3)

    temp_p_c_z = K.exp(K.sum(inner_part, axis=1)) + 1e-10

    return temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

def vae_loss(x, x_decoded):
    z_3 = tf.transpose(K.repeat(z, n_feat), [0, 2, 1])

    mu_3 = K.repeat_elements(tf.expand_dims(mu, 0), batch_size, axis=0)
    sigma_3 = K.repeat_elements(tf.expand_dims(sigma, 0), batch_size, axis=0)
    p_pi_3 = K.repeat_elements(tf.expand_dims(p_pi, 0), latent_size, axis=0)
    p_pi_3 = K.repeat_elements(tf.expand_dims(p_pi_3, 0), batch_size, axis=0)

    inner_part = K.log(p_pi_3) - 0.5 * K.log(2 * math.pi * sigma_3) - K.square(z_3 - mu_3) / (2 * sigma_3)

    temp_p_c_z = K.exp(K.sum(inner_part, axis=1)) + 1e-10
    gamma = temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

    gamma_t = K.repeat(gamma, latent_size)

    z_mean_t = tf.transpose(K.repeat(z_mean, n_feat), [0, 2, 1])
    z_log_var_t = tf.transpose(K.repeat(z_log_var, n_feat), [0, 2, 1])

    dec_error = 0.9 * input_dim * binary_crossentropy(x, x_decoded)

    dec_error += K.sum(0.5*gamma_t*(latent_size*K.log(math.pi*2)+K.log(sigma_3)+K.exp(z_log_var_t)/sigma_3+K.square(z_mean_t-mu_3)/sigma_3), axis=(1, 2))
    dec_error -= 0.5*K.sum(z_log_var+1, axis=-1)
    change = K.repeat_elements(tf.expand_dims(p_pi, 0), batch_size, 0)
    dec_error -= K.sum(K.log(change)*gamma, axis=-1)
    dec_error += K.sum(K.log(gamma)*gamma, axis=-1)

    return dec_error

# layer def
layer_sizes = [256, 256, 512]
latent_size = 10
n_feat = 10
input_dim = 784
batch_size = 128

x = Input(shape=(input_dim,))
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

Gamma = Lambda(calc_gamma, output_shape=(n_feat,))(z)
sample_output = Model(x, z_mean)
gamma_output = Model(x,Gamma)

GMM_model = Model(x, x_decoded_mean)

lr_nn = 0.002
lr_gmm = 0.002
adam_nn= Adam(lr=lr_nn, epsilon=1e-4)
adam_gmm= Adam(lr=lr_gmm, epsilon=1e-4)

GMM_model.compile(optimizer=adam_nn, loss=vae_loss,add_trainable_weights=[p_pi, mu, sigma],add_optimizer=adam_gmm)

train_values, train_labels, test_values, test_labels = get_small_test_batch(512)

GMM_model.fit(train_values, train_values,
        shuffle=True,
        epochs=50,
        batch_size=batch_size,
        validation_data=(test_values, test_values))
