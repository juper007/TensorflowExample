import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

nb_classes = 10;

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name='bais1')
layar1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layar1 = tf.nn.dropout(layar1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name='bais2')
layar2 = tf.nn.relu(tf.matmul(layar1, W2) + b2)
layar2 = tf.nn.dropout(layar2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]), name='bais3')
layar3 = tf.nn.relu(tf.matmul(layar2, W3) + b3)
layar3 = tf.nn.dropout(layar3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]), name='bais4')
hypothesis = tf.nn.softmax(tf.matmul(layar3, W4) + b4)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate=0.3).minimize(cost)

isCorrect = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

training_epoch = 500
batch_size = 100

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s, _ = sess.run([summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            writer.add_summary(s)

        print('%04d' % (epoch + 1), ",", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()