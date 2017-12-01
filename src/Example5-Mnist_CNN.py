import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from datetime import datetime
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

nb_classes = 10;

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, nb_classes])
# keep_prob = tf.placeholder(tf.float32)

# W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

L3 = tf.reshape(L3, [-1, 4*4*128])

W4 = tf.get_variable("W4", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]), name='bais4')
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

W5 = tf.get_variable("W5", shape=[625, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bais5')
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

isCorrect = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

training_epoch = 20
batch_size = 100

with tf.Session() as sess:
    logdir = './logs/CNN_' + datetime.now().strftime('%Y%m%d%H%M')
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, s, _ = sess.run([cost, summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            # c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch
            writer.add_summary(s, global_step=i + (epoch*total_batch))

        print('%04d' % (epoch + 1), ",", '{:.9f}'.format(avg_cost), ",", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()