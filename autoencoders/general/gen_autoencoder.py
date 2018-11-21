from keras.models import Model
from keras.layers import Dense, Input
from time import gmtime, strftime
from get_mnist import get_number_mnist

train_values, train_labels, test_values, test_labels = get_number_mnist()
# to 28x28
# train_values = train_values.reshape(train_values.shape[0], 28, 28, 1)
# test_values = test_values.reshape(test_values.shape[0], 28, 28, 1)

compress_dim = 32

input_img = Input(shape=(784,))

encoded = Dense(compress_dim, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train_values, train_values,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(test_values, test_values))

autoencoder.save(strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.h5')