import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist


class GAN:
    def __init__(self, raw_images_data):
        tf.reset_default_graph()
        # model parameters
        self.Z_DIM = 100
        self.X_DIM = 28
        self.BATCH_SIZE = 16
        self.TRAIN_ITERS = 3000
        self.optimizer = tf.train.AdamOptimizer()

        self.train_writer = None

        self.iterator = None
        self._prep_data(raw_images_data)

        # Placeholder for input images to the discriminator
        self.x_placeholder = tf.placeholder(
            "float", shape=[None, self.X_DIM, self.X_DIM, 1])
        # Placeholder for input noise vectors to the generator
        self.z_placeholder = tf.placeholder(
            tf.float32, [None, self.Z_DIM])

    def _prep_data(self, raw_images_data):
        # TODO: IMPLEMENT THIS WITH A NEW DATA SET
        dataset = tf.data.Dataset.from_tensor_slices(raw_images_data)
        self.iterator = dataset.repeat().batch(self.BATCH_SIZE).make_one_shot_iterator()

        # TODO: IMPLEMENT CYCLEGAN?

    def train(self):
        # TensorBoard
        self.merged = tf.summary.merge_all()

        # retrieve graph
        g_loss, d_loss = self._get_losses()
        trainerD, trainerG = self._get_trainers(g_loss, d_loss)

        # initialize graph
        with tf.Session() as sess:
            # TensorBoard
            self.train_writer = tf.summary.FileWriter('tensorboard/train',
                                                      sess.graph)
            sess.run(tf.global_variables_initializer())
            next_el = self.iterator.get_next()
            for i in range(self.TRAIN_ITERS):
                # generate inputs
                z_batch = np.random.uniform(-1, 1,
                                            size=[self.BATCH_SIZE, self.Z_DIM])
                real_image_batch = sess.run(next_el)
                real_image_batch = np.reshape(
                    real_image_batch, [self.BATCH_SIZE, self.X_DIM, self.X_DIM, 1])

                # run through net
                _, dLoss, summary = sess.run([trainerD, d_loss, self.merged], feed_dict={
                    self.z_placeholder: z_batch, self.x_placeholder: real_image_batch})  # Update the discriminator
                _, gLoss = sess.run([trainerG, g_loss], feed_dict={
                                    self.z_placeholder: z_batch})  # Update the generator
                self.train_writer.add_summary(summary, i)

                # print update
                if i % 200 == 0:
                    print('dLoss: %f\ngLoss: %f\n' % (dLoss, gLoss))
                if i == 5:
                    print('training going ok')

    def _get_losses(self):
        # Dx will hold discriminator prediction probabilities for the real MNIST images
        Dx = self._discriminator(self.x_placeholder)
        # Gz holds the generated images
        Gz = self._generator(self.z_placeholder,
                             self.BATCH_SIZE, self.Z_DIM)
        # Dg will hold discriminator prediction probabilities for generated images
        Dg = self._discriminator(Gz, reuse=True)

        # ensure forward compatibility: function needs to have logits and labels args explicitly used
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg, labels=tf.ones_like(Dg)))

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dx, labels=tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg, labels=tf.zeros_like(Dg)))
        d_loss = d_loss_real + d_loss_fake

        return g_loss, d_loss

    def _get_trainers(self, g_loss, d_loss):
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

        with tf.variable_scope('d_train'):
            trainerD = self.optimizer.minimize(d_loss, var_list=d_vars)
        with tf.variable_scope('g_train'):
            trainerG = self.optimizer.minimize(g_loss, var_list=g_vars)
        return trainerD, trainerG

    def _variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            # tf.summary.scalar('mean', mean)
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _d_conv_layer(layer, filter):
        W_conv = tf.get_variable(
            'd_weights_conv', filter, initializer=tf.truncated_normal_initializer(stddev=.01))
        b_conv = tf.get_variable(
            'd_bias_conv', [filter[-1]], initializer=tf.constant_initializer(0))

        GAN._variable_summaries(W_conv)
        GAN._variable_summaries(b_conv)

        layer = tf.nn.conv2d(input=layer, filter=W_conv, strides=[
                             1, 1, 1, 1], padding='SAME')
        layer = tf.nn.relu(layer + b_conv)
        layer = tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')
        return layer

    def _d_fc_layer(layer, in_dim, out_dim, last):
        W_fc = tf.get_variable(
            'd_weights_fc', [in_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=.01))
        b_fc = tf.get_variable(
            'b_fc', [out_dim], initializer=tf.constant_initializer(0))
        layer = tf.matmul(layer, W_fc) + b_fc
        if not last:
            layer = tf.nn.relu(layer)
        return layer

    def _discriminator(self, x_image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            # First Conv and Pool Layers
            with tf.variable_scope('conv1'):
                layer = GAN._d_conv_layer(x_image, [5, 5, 1, 8])

            # Second Conv and Pool Layers
            with tf.variable_scope('conv2'):
                layer = GAN._d_conv_layer(layer, [5, 5, 8, 16])

            # First Fully Connected Layer
            layer = tf.reshape(layer, [-1, 7 * 7 * 16])
            with tf.variable_scope('fc1'):
                layer = GAN._d_fc_layer(layer, 7 * 7 * 16, 32, False)

            # Second (Final) Fully Connected Layer
            with tf.variable_scope('fc2'):
                y_conv = GAN._d_fc_layer(layer, 32, 1, True)

        return y_conv

    def _g_deconv_layer(layer, output_shape, scope):
        W_conv = tf.get_variable('g_weights_conv',
                                 [5, 5, output_shape[-1],
                                  int(layer.get_shape()[-1])],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv = tf.get_variable('g_bias_conv',
                                 [output_shape[-1]],
                                 initializer=tf.constant_initializer(.1))

        GAN._variable_summaries(W_conv)
        GAN._variable_summaries(b_conv)

        layer = tf.nn.conv2d_transpose(layer, W_conv, output_shape=output_shape,
                                       strides=[1, 2, 2, 1], padding='SAME') + b_conv
        layer = tf.contrib.layers.batch_norm(
            inputs=layer, center=True, scale=True, is_training=True, scope=scope)
        layer = tf.nn.relu(layer)
        return layer

    def _generator(self, z, batch_size, z_dim, reuse=False):
        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()
            g_dim = 64  # Number of filters of first layer of generator
            c_dim = 1  # 1 color channel in output image
            s = 28  # dimension of output image
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

            layer = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 25])
            layer = tf.nn.relu(layer)
            # Dimensions = batch_size x 2 x 2 x 25

            # First DeConv Layer
            output1_shape = [batch_size, s8, s8, g_dim * 4]
            with tf.variable_scope('conv1'):
                layer = GAN._g_deconv_layer(
                    layer, output1_shape, 'g_bn1')
            # Dimensions = batch_size x 3 x 3 x 256

            # Second DeConv Layer
            output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
            with tf.variable_scope('conv2'):
                layer = GAN._g_deconv_layer(
                    layer, output2_shape, 'g_bn2')
            # Dimensions = batch_size x 6 x 6 x 128

            # Third DeConv Layer
            output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
            with tf.variable_scope('conv3'):
                layer = GAN._g_deconv_layer(
                    layer, output3_shape, 'g_bn3')
            # Dimensions = batch_size x 12 x 12 x 64

            # Fourth DeConv Layer
            output4_shape = [batch_size, s, s, c_dim]
            with tf.variable_scope('conv4'):
                W_conv4 = tf.get_variable('g_weights_conv',
                                          [5, 5, output4_shape[-1],
                                           int(layer.get_shape()[-1])],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_conv4 = tf.get_variable('g_bias_conv',
                                          [output4_shape[-1]],
                                          initializer=tf.constant_initializer(.1))
                layer = tf.nn.conv2d_transpose(layer, W_conv4, output_shape=output4_shape,
                                               strides=[1, 2, 2, 1], padding='VALID') + b_conv4
                layer = tf.nn.tanh(layer)
            # Dimensions = batch_size x 28 x 28 x 1

        return layer


if __name__ == "__main__":
    # import data
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/")

    train_images, _ = loadlocal_mnist(
        'MNIST_data/t10k-images-idx3-ubyte', 'MNIST_data/t10k-labels-idx1-ubyte')
    print('SHAPE IS ', train_images.shape)

    # run model
    model = GAN(train_images)
    model.train()
