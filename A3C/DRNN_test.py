import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import random

random.seed(0)
np.random.seed(0)
seq_max_len = 10
lr = 1e-3
total_episodes = 10
batch_size = 5
num_units = 10


class Env:
    def __init__(self):
        self.total_reward = 0
        self.sample = []

    def generate_sample(self):
        self.sample = [np.random.normal(0, 1, 2) for x in range(seq_max_len)]

    def step(self, actions):
        for i in range(len(actions)-1):
            self.total_reward += LA.norm(self.sample[actions[i]] - self.sample[actions[i+1]])


class Data:
    def __init__(self, batch_size, seq_max_len):
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len

    def generate_batch(self):
        s = []
        A = []
        G = []
        for i in range(self.batch_size):
            actions = [i for i in range(self.seq_max_len)]
            random.shuffle(actions)
            env = Env()
            env.generate_sample()
            env.step(actions)
            s.append(env.sample)
            A.append(actions)
            G.append(env.total_reward)
        return np.asarray(s, dtype=np.float32), np.asarray(A, dtype=np.float32), np.asarray(G, dtype=np.float32)


init = tf.global_variables_initializer()
data = Data(batch_size, seq_max_len)
s_batch, A_batch, G_batch = data.generate_batch()
s_batch = s_batch[:,:,0]
s_batch = s_batch.reshape(-1, 10, 1)
print()


# input_embed = tf.get_variable(
#         "input_embed", [1, 2, 2],
#         initializer=tf.random_uniform_initializer(1, 1.01))
# input_filter = tf.ones([1, 2, 2])
# tmp = tf.nn.conv1d(s_batch, input_filter, 1, "VALID")
s = tf.placeholder(tf.float32, [None, seq_max_len, 1])
# A = tf.placeholder(tf.float32, [None, seq_max_len])
# G = tf.placeholder(tf.float32, [None, 1])
# seqlen = tf.placeholder(tf.int32, [None])

# tmp1 = tf.contrib.rnn.BasicRNNCell(10)
# tmp1 = tf.contrib.rnn.OutputProjectionWrapper(tmp1, output_size=1)
# tmp, _ = tf.nn.dynamic_rnn(tmp1, s, dtype=tf.float32)


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):

        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))
        # Create batch Time Series on t axis
        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution
        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)
        # Format for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)
batch_size = 1
num_time_steps = 10
num_inputs = 1
num_neurons = 7
num_outputs = 1
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    X_batch, _ = ts_data.next_batch(batch_size, num_time_steps)
    print(sess.run(outputs, feed_dict={X: X_batch}))
# encoder_cell = tf.contrib.rnn.GridLSTMCell(num_units, [t])
# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
#     encoder_cell,
#     s,
#     # sequence_length=np.repeat([seq_max_len], batch_size),
#     dtype=tf.float32)            # Not sure about time major
# print(encoder_state)
# # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# # decoder_outputs, decoder_state = tf.nn.dynamic_rnn(
# #     decoder_cell,
# #     encoder_state,
# #     sequence_length=np.repeat([seq_max_len], batch_size),
# #     time_major=False,
# #     dtype=tf.float32)            # Not sure about time major


# with tf.Session() as sess:
#     # feed_dict = {s: s_batch, A: A_batch, G: G_batch}
#     sess.run(init)
#     feed_dict = {s: s_batch}
#     # sess.run(encoder_outputs, feed_dict=feed_dict)
#     out = sess.run(tmp, feed_dict=feed_dict)
#     print(out.shape)


# s_in = tf.unstack(s, seq_max_len, 1)
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(10)
# rnn_outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, s_in, dtype=tf.float32, sequence_length=seqlen)
# output = tf.layers.dense(inputs=rnn_outputs, units=10, activation=None)

# neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=A_t)
# loss = tf.reduce_mean(neg_log_prob * G_t)

# trainable_variables = tf.trainable_variables()
# gradient_holders = []

# for idx, var in enumerate(trainable_variables):
#     placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
#     gradient_holders.append(placeholder)

# gradients = tf.gradients(loss, trainable_variables)

# optimizer = tf.train.AdamOptimizer(lr)
# update_batch = optimizer.apply_gradients(zip(gradient_holders, trainable_variables))

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)

#     i = 0
#     gradBuffer = sess.run(tf.trainable_variables())
#     for ix, grad in enumerate(gradBuffer):
#         gradBuffer[ix] = grad*0

#     while i < total_episodes:
#         i += 1
#         # s_in = np.random.uniform(0, 1, N).reshape(-1, 1)
#         # G_t, A_t = reward_to_end(s_in, t)
#         # # print(A_t, G_t)
#         # s_in = np.array([s_in[t]], dtype=np.float32)
#         # r_in = np.array([G_t], dtype=np.float32)
#         # a_in = np.array([A_t], dtype=np.int32)
#         # feed_dict = {s: s_in, reward_holder: r_in, action_holder: a_in}
#         # grads = sess.run(gradients, feed_dict=feed_dict)
#         # for idx, grad in enumerate(grads):
#         #     gradBuffer[idx] += grad
#         # feed_dict = dict(zip(gradient_holders, gradBuffer))
#         # _ = sess.run(update_batch, feed_dict=feed_dict)


# test = model()
# test.generate_sample()
# test.step(range(3))
# print(test.sample)
# print(test.total_reward)


# SOLVE THE TSP PROBLEM
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
# def solve_tsp_dynamic(points):
#   #calc all lengths
#   all_distances = [[length(x,y) for y in points] for x in points]
#   #initial value - just distance from 0 to every other point + keep the track of edges
#   A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
#   cnt = len(points)
#   for m in range(2, cnt):
#     B = {}
#     for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
#       for j in S - {0}:
#         B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
#     A = B
#   res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
#   return np.asarray(res[1]) + 1 # 0 for padding
