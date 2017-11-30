import tensorflow as tf

X = [1, 2, 3]
Y = [2.1, 3.1, 4.1]

W = tf.Variable(5.)
b = tf.Variable(3.)

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    print(step, sess.run([W, b]))
    sess.run(train)