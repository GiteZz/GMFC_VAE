import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_sample = mnist.train.num_examples
input_dim = mnist.train.images[0].shape[0]
w = h = 28
batch_size = 100

class VariantionalAutoencoder(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, n_z=10):
        self.layer_sizes = [500, 500, 2000]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[self.batch_size, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        self.f1 = fc(self.x, self.layer_sizes[0], scope='enc_fc1', activation_fn=tf.nn.relu)
        self.f2 = fc(self.f1, self.layer_sizes[1], scope='enc_fc2', activation_fn=tf.nn.relu)
        self.f3 = fc(self.f2, self.layer_sizes[2], scope='enc_fc3', activation_fn=tf.nn.relu)

        self.z_mu = fc(self.f3, self.n_z, scope='enc_fc4_mu', activation_fn=None)
        self.z_sigma = fc(self.f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_sigma), mean=0, stddev=1, dtype=tf.float32)

        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_sigma)) * eps

        # Decode
        # z -> x_hat
        self.g1 = fc(self.z, self.layer_sizes[-1], scope='dec_fc1', activation_fn=tf.nn.relu)
        self.g2 = fc(self.g1, self.layer_sizes[-2], scope='dec_fc2', activation_fn=tf.nn.relu)
        self.g3 = fc(self.g2, self.layer_sizes[-3], scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(self.g3, input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_sigma - tf.square(self.z_mu) - tf.exp(self.z_sigma), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )
        return loss, recon_loss, latent_loss

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

    def save_weights(self):
        save_dict = {}
        all_vars = tf.global_variables()

        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None

        save_dict['enc_fc1_w'] = get_var('enc_fc1/weights')
        save_dict['enc_fc1_b'] = get_var('enc_fc1/biases')
        save_dict['enc_fc2_w'] = get_var('enc_fc2/weights')
        save_dict['enc_fc2_b'] = get_var('enc_fc2/biases')
        save_dict['enc_fc3_w'] = get_var('enc_fc3/weights')
        save_dict['enc_fc3_b'] = get_var('enc_fc3/biases')

        save_dict['z_mu_w'] = get_var('enc_fc4_mu/weights')
        save_dict['z_mu_b'] = get_var('enc_fc4_mu/biases')
        save_dict['z_sigma_w'] = get_var('enc_fc4_sigma/weights')
        save_dict['z_sigma_b'] = get_var('enc_fc4_sigma/biases')

        save_dict['dec_fc1_w'] = get_var('dec_fc1/weights')
        save_dict['dec_fc1_b'] = get_var('dec_fc1/biases')
        save_dict['dec_fc2_w'] = get_var('dec_fc2/weights')
        save_dict['dec_fc2_b'] = get_var('dec_fc2/biases')
        save_dict['dec_fc3_w'] = get_var('dec_fc3/weights')
        save_dict['dec_fc3_b'] = get_var('dec_fc3/biases')

        save_dict['dec_fc4_w'] = get_var('dec_fc4/weights')
        save_dict['dec_fc4_b'] = get_var('dec_fc4/biases')

        saver = tf.train.Saver(save_dict)
        saver.save(self.sess, "./pretrain/pretrain_mnist.ckpt")


def trainer(learning_rate=1e-3, batch_size=100, num_epoch=75, n_z=10):
    model = VariantionalAutoencoder(learning_rate=learning_rate,
                                    batch_size=batch_size, n_z=n_z)

    for epoch in range(num_epoch):
        for iter in range(num_sample // batch_size):
            # Obtina a batch
            batch = mnist.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            loss, recon_loss, latent_loss = model.run_single_step(batch[0])

        print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(epoch, loss, recon_loss, latent_loss))

    print('Done!')
    return model

# Train the model
model = trainer(learning_rate=1e-3,  batch_size=100, num_epoch=10, n_z=2)

model.save_weights()

batches = np.empty((0, 784))
numbers = np.empty((0, 10))
z_s = np.empty((0, 2))

# Test the trained model: transformation
for i in range(30):
    batch = mnist.test.next_batch(batch_size)
    z = model.transformer(batch[0])
    z_s = np.concatenate((z_s, z))
    batches = np.concatenate((batches, batch[0]))
    numbers = np.concatenate((numbers, batch[1]))

fig = plt.figure()
plt.scatter(z_s[:, 0], z_s[:, 1], c=np.argmax(numbers, 1))
plt.colorbar()
plt.grid()
plt.savefig('results/pretrain.png')
plt.close(fig)