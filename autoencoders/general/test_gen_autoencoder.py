from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Dense, Input
import numpy as np
import pandas as pd
from time import gmtime, strftime
import matplotlib.pyplot as plt
from get_mnist import get_number_mnist, get_small_test_batch

train_values, train_labels, test_values, test_labels = get_small_test_batch(10)
model = load_model('2018_11_20_20_24_37.h5')

# construct encoder based on full autoencoder
encoder_input = Input(shape=(784,))
encoder_output = model.layers[1](encoder_input)

encoder = Model(inputs=encoder_input, output=encoder_output)

# construct decoder based on full autoencoder
decoder_input = Input(shape=(32,))
decoder_output = model.layers[2](decoder_input)

decoder = Model(decoder_input, decoder_output)

encoded_imgs = encoder.predict(test_values)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_values[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()