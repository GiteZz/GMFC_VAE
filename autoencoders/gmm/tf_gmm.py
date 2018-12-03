import tensorflow as tf
from get_mnist import get_number_mnist
from time import gmtime, strftime

layer_sizes = [512, 512, 2048]
latent_size = 10
n_feat = 10
input_dim = 784
batch_size = 128

class VariationalAutoencoder(object):

    def __init__(self, learning_rate=1e-3, batch_size=128, n_z=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        # Encoder
        self.x = tf.placeholder(tf.float32, shape=(None, input_dim))
        I1 = tf.layers.dense(self.x, layer_sizes[0], tf.nn.relu)
        I2 = tf.layers.dense(I1, layer_sizes[1], tf.nn.relu)
        I3 = tf.layers.dense(I2, layer_sizes[2], tf.nn.relu)

        # Latent space
        self.z_mean = tf.layers.dense(I3, latent_size, tf.nn.relu)
        self.z_var = tf.layers.dense(I3, latent_size, tf.nn.relu)

        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1, dtype=tf.float32)

        self.z = self.z_mean + eps * self.z_var

        # Decoder
        D1 = tf.layers.dense(self.z, layer_sizes[2], tf.nn.relu)
        D2 = tf.layers.dense(D1, layer_sizes[1], tf.nn.relu)
        self.x_dec = tf.layers.dense(D2, layer_sizes[0], tf.nn.sigmoid)

        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon + self.x_dec) + (1 - self.x) * tf.log(epsilon + 1 - self.x_dec),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: difference between two distributions
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_var - tf.square(self.z_mean) - tf.exp(self.z_var), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

        return

