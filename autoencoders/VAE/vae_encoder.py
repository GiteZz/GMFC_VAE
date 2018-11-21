'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from get_mnist import get_number_mnist
from time import gmtime, strftime

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

input_dim = 784
inter_dim = 64
latent_dim = 32



# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_loss(x, x_decoded_mean):
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + vae_stdev_layer - K.square(vae_latent_layer) - K.exp(vae_stdev_layer), axis=-1)
    return xent_loss + kl_loss

# create all layers
vae_en_input = Input(shape=(input_dim,))
vae_en_inter = Dense(inter_dim, activation='relu')
vae_mean_latent = Dense(latent_dim, activation='relu')
vae_stdev_latent = Dense(latent_dim, activation='relu')
vae_latent = Lambda(sampling, output_shape=(latent_dim,))
vae_de_inter = Dense(inter_dim, activation='relu')
vae_de_output = Dense(input_dim, activation='sigmoid')


# create VAE
vae_en_inter_layer = vae_en_inter(vae_en_input)
vae_mean_layer = vae_mean_latent(vae_en_inter_layer)
vae_stdev_layer = vae_stdev_latent(vae_en_inter_layer)
vae_latent_layer = vae_latent([vae_mean_layer, vae_stdev_layer])
vae_de_inter_layer = vae_de_inter(vae_latent_layer)
vae_de_output_layer = vae_de_output(vae_de_inter_layer)

vae_model = Model(vae_en_input, vae_de_output_layer)

#create encoder
encoder_model = Model(vae_en_input, [vae_mean_layer, vae_stdev_layer])

# create decoder
dec_input = Input(shape=(latent_dim,))
dec_inter_layer = vae_de_inter(dec_input)
dec_output_layer = vae_de_output(dec_inter_layer)

decoder_model = Model(dec_input, dec_output_layer)

# compile and train
vae_model.compile(optimizer='rmsprop', loss=vae_loss)


train_values, train_labels, test_values, test_labels = get_number_mnist()

vae_model.fit(train_values, train_values,
        shuffle=True,
        epochs=50,
        batch_size=128,
        validation_data=(test_values, test_values))

vae_model.save('VAE' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.h5')
encoder_model.save('encoder' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.h5')
decoder_model.save('decoder' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.h5')

