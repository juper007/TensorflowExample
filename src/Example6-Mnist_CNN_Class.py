import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

learning_rate = 0.001
training_epoch = 20
batch_size = 100

class Model:
    def __init__(self, sess, name):
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2, "SAME")
            dropout1 = tf.layers.dropout(inputs=pool1, rate=self.keep_prob, training=self.training)

            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2, "SAME")
            dropout2 = tf.layers.dropout(pool2, self.keep_prob, training=self.training)

            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, [2, 2], 2, "SAME")
            dropout3 = tf.layers.dropout(inputs=pool3, rate=self.keep_prob, training=self.training)

            flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])

            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())
            dropout4 = tf.layers.dropout(inputs=dense4, rate=self.keep_prob, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout4, units=10,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer())

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            cost_summ = tf.summary.scalar("cost", self.cost)
            self.summary = tf.summary.merge_all()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.keep_prob: keep_prob, self.training: training})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob, self.training: training})

    def train(self, x_data, y_data, keep_prob=0.5, training=True):
        return self.sess.run([self.summary, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob, self.training: training})


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
m1 = Model(sess, "m1")

logdir = './logs/CNN_' + datetime.now().strftime('%Y%m%d%H%M')
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())


for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        s, _ = m1.train(batch_xs, batch_ys)
        writer.add_summary(s, global_step=i + (epoch*total_batch))

print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
