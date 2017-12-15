import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_date/")


def discriminator(images, reuse=False):
    with tf.variable_scope("Generator"):
        conv1 = tf.layers.conv2d(name='d_conv1',
                                 inputs=images,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 strides=[1, 1],
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 1, "SAME")

        conv2 = tf.layers.conv2d(name='d_conv2',
                                 inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 strides=[1, 1],
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
        pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 1, "SAME")

        dense3 = tf.layers.dense(name='d_dense3',
                                 inputs=pool2,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)

        dense4 = tf.layers.dense(name='d_dense4',
                                 inputs=dense3,
                                 units=1,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
    return dense4


def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope("Generator"):
        dense1 = tf.layers.dense(name='g_dense1',
                                 inputs=z,
                                 units=56 * 56,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
        dense1 = tf.reshape(dense1, [-1, 56, 56, 1])

        conv2 = tf.layers.conv2d(name='g_conv2',
                                 inputs=dense1,
                                 filters=z_dim/2,
                                 kernel_size=[3, 3],
                                 strides=[2, 2],
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
        conv2 = tf.image.resize_images(conv2, [56, 56])

        conv3 = tf.layers.conv2d(name='g_conv3',
                                 inputs=conv2,
                                 filters=z_dim/4,
                                 kernel_size=[3, 3],
                                 strides=[2, 2],
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
        conv3 = tf.image.resize_images(conv3, [56, 56])

        conv4 = tf.layers.conv2d(name='g_conv4',
                                 inputs=conv3,
                                 filters=1,
                                 kernel_size=[1, 1],
                                 strides=[2, 2],
                                 padding="SAME",
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0),
                                 reuse=reuse)
    return conv4


z_dimensions = 100
batch_size = 50

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name="z_placeholder")
x_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x_placeholder")

z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

Gz = generator(z_placeholder, batch_size, z_dimensions)
Dx = discriminator(x_placeholder)
Dg = discriminator(Gz, True)


d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

d_trainer_fake = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(d_loss_real, var_list=d_vars)

g_trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss, var_list=g_vars)


tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     generated_image = sess.run(Gz, feed_dict={z_placeholder: z_batch})
#     generated_image = generated_image.reshape([28, 28])
#     plt.imshow(generated_image, cmap='Greys')
#     plt.show()


with tf.Session() as sess:
    images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions, True)
    tf.summary.image('Generated_image', images_for_tensorboard, 5)
    merged = tf.summary.merge_all()
    logdir = 'GAN_log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    writer = tf.summary.FileWriter(logdir, sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(300):
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        real_image = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                              feed_dict={x_placeholder: real_image, z_placeholder: z_batch})

        print("dLossReal", dLossReal, "dLossFake", dLossFake)

    for i in range(100000):
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        real_image = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                              feed_dict={x_placeholder: real_image, z_placeholder: z_batch})

        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

        if i % 10 == 0:
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            summary = sess.run(merged, feed_dict={x_placeholder: real_image, z_placeholder: z_batch})
            writer.add_summary(summary, i)

        if i % 100 == 0:
            print("Iteration:", i, "at", datetime.datetime.now())
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            generated_images = generator(z_placeholder, 1, z_dimensions, True)
            images = sess.run(generated_images, {z_placeholder: z_batch})
            plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
            # plt.show()

























