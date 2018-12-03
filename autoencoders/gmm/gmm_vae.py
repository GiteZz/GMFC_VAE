import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import logging

logger = logging.getLogger('GMMM_VAE')

# from get_mnist import get_number_mnist, get_small_test_batch
#
# train_values, train_labels, test_values, test_labels = get_small_test_batch(10)
# input_dim = 784

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_sample = mnist.train.num_examples
input_dim = mnist.train.images[0].shape[0]
w = h = 28

class VariantionalAutoencoder(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, latent_size=10, gmm_n_feat=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.gmm_pi = tf.Variable(tf.ones(gmm_n_feat) / gmm_n_feat, trainable=True, dtype=tf.float32)
        self.gmm_mu = tf.Variable(tf.zeros(gmm_n_feat), trainable=True, dtype=tf.float32)
        self.gmm_sigma = tf.Variable(tf.ones(gmm_n_feat), trainable=True, dtype=tf.float32)
        self.gmm_n_feat = gmm_n_feat
        print('start build')
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        amount = len(tf.trainable_variables())
        print("amount of trainable vars: " + str(amount))

    # Build the network and the loss functions
    def build(self):
        self.debug_dict = {}
        self.debug_dict['gmm_pi'] = self.gmm_pi
        self.debug_dict['gmm_mu'] = self.gmm_mu
        self.debug_dict['gmm_sigma'] = self.gmm_sigma

        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[self.batch_size, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.relu)

        self.z_mu = fc(f3, self.latent_size, scope='enc_fc4_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.latent_size, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)

        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 256, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss, cross entropy
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
            axis=1
        )
        recon_loss = tf.reduce_mean(recon_loss)

        # change to shape (batch_dim, latent_dim, n_feat)
        gmm_mu_3 = tf.ones([self.batch_size, self.latent_size, self.gmm_n_feat]) * self.gmm_mu
        gmm_pi_3 = tf.ones([self.batch_size, self.latent_size, self.gmm_n_feat]) * self.gmm_pi
        gmm_pi_2 = tf.ones([self.batch_size, self.gmm_n_feat]) * self.gmm_pi
        gmm_sigma_3 = tf.ones([self.batch_size, self.latent_size, self.gmm_n_feat]) * self.gmm_sigma

        # Now make sure the network output are also that shape
        z_3 = tf.tile(tf.expand_dims(self.z, -1), [1, 1, self.gmm_n_feat])
        mu_3 = tf.tile(tf.expand_dims(self.z_mu, -1), [1, 1, self.gmm_n_feat])
        var_3 = tf.tile(tf.expand_dims(self.z_log_sigma_sq, -1), [1, 1, self.gmm_n_feat])

        # shape = (batch, n_feat)
        p_c_z = tf.exp(tf.reduce_sum(tf.log(gmm_pi_3) - 0.5 * tf.log(2 * math.pi * gmm_sigma_3) - tf.square(z_3 - gmm_mu_3)/ (2 * gmm_sigma_3), 1)) + 1e-10
        # same shape because broadcasting
        gamma = p_c_z / tf.reduce_sum(p_c_z, -1, keepdims=True)

        gamma_3 = tf.tile(tf.expand_dims(gamma, 1), [1, self.latent_size, 1])


        # these should all be shape=(batch_dim)
        loss1 = tf.reduce_sum(0.5 * gamma_3 * (self.latent_size * tf.log(math.pi * 2) + tf.log(gmm_sigma_3) + tf.exp(var_3) / gmm_sigma_3 + tf.square(mu_3 - gmm_mu_3) / gmm_sigma_3), [1, 2])
        loss2 = -0.5 * tf.reduce_sum(self.z_log_sigma_sq + 1, -1)
        loss3 = -1 * tf.reduce_sum(tf.log(gmm_pi_2) * gamma, -1)
        loss4 = tf.reduce_sum(tf.log(gamma) * gamma, -1)

        # create the gmm loss with shape = (batch) and then the gmm_loss as just a value (reduce_mean)

        gmm_loss = tf.reduce_mean(loss1 + loss2 + loss3 + loss4)

        self.total_loss = recon_loss + gmm_loss
        self.debug_dict['recon_loss'] = recon_loss
        self.debug_dict['gmm_loss'] = gmm_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, single_batch):
        _, debug_dict = self.sess.run(
            [self.train_op, self.debug_dict],
            feed_dict={self.x: single_batch}
        )
        return debug_dict

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


def trainer(learning_rate=2e-3, batch_size=100, num_epoch=75, latent_size=10):
    model = VariantionalAutoencoder(learning_rate=learning_rate,
                                    batch_size=batch_size, latent_size=latent_size, gmm_n_feat=10)

    # batches = np.split(train_values, 5)
    # for i in range(5):
    #     debug_dict = model.run_single_step(batches[i])
    #     print("============================= test {} =============================".format(i))
    #     for k, v in debug_dict.items():
    #         print(k)
    #         print(v)


    for epoch in range(num_epoch):
        for iter in range(num_sample // batch_size):
            # Obtina a batch
            batch = mnist.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            debug_dict = model.run_single_step(batch[0])

        if epoch % 5 == 0:
            print("======={}=======".format(epoch))
            for k, v in debug_dict.items():
                print(k)
                print(v)

    print('Done!')
    return model

print("start training")
# Train the model
model = trainer(learning_rate=1e-4,  batch_size=15, num_epoch=100, latent_size=10)

print("ended training")












# Test the trained model: reconstruction
batch = mnist.test.next_batch(100)
x_reconstructed = model.reconstructor(batch[0])

n = np.sqrt(model.batch_size).astype(np.int32)
I_reconstructed = np.empty((h*n, 2*w*n))
for i in range(n):
    for j in range(n):
        x = np.concatenate(
            (x_reconstructed[i*n+j, :].reshape(h, w),
             batch[0][i*n+j, :].reshape(h, w)),
            axis=1
        )
        I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

fig = plt.figure()
plt.imshow(I_reconstructed, cmap='gray')
plt.savefig('I_reconstructed.png')
plt.close(fig)

# Test the trained model: generation
# Sample noise vectors from N(0, 1)
z = np.random.normal(size=[model.batch_size, model.latent_size])
x_generated = model.generator(z)

n = np.sqrt(model.batch_size).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)

fig = plt.figure()
plt.imshow(I_generated, cmap='gray')
plt.savefig('I_generated.png')
plt.close(fig)

tf.reset_default_graph()
# Train the model with 2d latent space
model_2d = trainer(learning_rate=1e-4,  batch_size=100, num_epoch=50, latent_size=2)

# Test the trained model: transformation
batch = mnist.test.next_batch(3000)
z = model_2d.transformer(batch[0])
fig = plt.figure()
plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1))
plt.colorbar()
plt.grid()
plt.savefig('I_transformed.png')
plt.close(fig)

# Test the trained model: transformation
n = 20
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)

I_latent = np.empty((h*n, w*n))
for i, yi in enumerate(x):
    for j, xi in enumerate(y):
        z = np.array([[xi, yi]]*model_2d.batch_size)
        x_hat = model_2d.generator(z)
        I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

fig = plt.figure()
plt.imshow(I_latent, cmap="gray")
plt.savefig('I_latent.png')
plt.close(fig)