from collections import deque
import tensorflow as tf
import random
import numpy as np
import gym

env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, 'gym-results/', force=True)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000


class DQN:
    def __init__(self, sess, input_size, output_size, name="main"):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=100, l_rate=0.0015):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, input_size], "input-x")
            self._Y = tf.placeholder(tf.float32, [None, output_size], "output_y")

            W1 = tf.get_variable("W1", shape=[input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[h_size], initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.nn.relu(tf.matmul(self._X, W1) + b1)

            W2 = tf.get_variable("W2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=[h_size], initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

            W3 = tf.get_variable("W3", shape=[h_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", shape=[output_size], initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(L2, W3) + b3

            self._loss = tf.reduce_sum(tf.square(self._Y - self._Qpred))
            self._train = tf.train.AdamOptimizer(l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.sess.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.sess.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})


def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return DQN.update(x_stack, y_stack)


def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break


def main():
    num_episodes = 500
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")

        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        sess.run(copy_ops)

        for i in range(num_episodes):
            e = 1. / ((i / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break

            print("Episode: {} Steps: {}".format(i, step_count))

            if i % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                sess.run(copy_ops)

        bot_play(mainDQN)


if __name__ == '__main__':
    main()
