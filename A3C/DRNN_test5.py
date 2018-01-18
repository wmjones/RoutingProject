import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
# seq_max_len = 10
# lr = 1e-3
# total_episodes = 10
# batch_size = 5
# num_units = 10


# class TSP_data():
#     def __init__(self, num_of_nodes):
#         self.num_of_nodes = num_of_nodes

#     def next_batch(self, batch_size=batch_size):
#         return(np.random.rand(batch_size, self.num_of_nodes, 2))


class TimeSeriesData():
    def __init__(self, num_points, num_buckets):

        self.num_buckets = num_buckets
        self.num_points = num_points
        self.t_data = np.linspace(0, 1, num_points)
        self.x_data = [[self.t_data*np.cos(self.t_data[i]*2*np.pi), self.t_data[i]*np.sin(self.t_data[i]*2*np.pi)]
                       for i in range(len(self.t_data))]

    def ret_true(self, t_series):
        return [[t_series[i]*np.cos(t_series[i]*2*np.pi), t_series[i]*np.sin(t_series[i]*2*np.pi)]
                for i in range(len(t_series))]

    def next_batch(self, batch_size, steps_in, steps_out):
        rand_start = np.random.rand(batch_size, 1)/2
        batch_ts = rand_start + np.arange(0.0, steps_in + steps_out)/self.num_points
        batch = np.asarray([self.ret_true(batch_ts[i]) for i in range(len(batch_ts))])
        batch_in = batch[:, :-steps_out, :]
        batch_out = batch[:, steps_in:, :]
        return(batch_in, batch_out)


num_buckets = 4
ts_data = TimeSeriesData(20, num_buckets)
batch_size = 10
hidden_dim = 30
output_dim = input_dim = 2
layers_stacked_count = 2
steps_in = 10
steps_out = 20

sample_x, sample_y = ts_data.next_batch(batch_size, steps_in, steps_out)
seq_length = sample_x.shape[1]
enc_inp = tf.placeholder(tf.float32, [None, steps_in, 2])
expected_sparse_output = tf.placeholder(tf.float32, [None, steps_out, 2])
keep_prob = tf.placeholder(tf.float32)

output_lengths = np.asarray([steps_out]*batch_size, dtype=np.int32)
in_cells = []
for i in range(layers_stacked_count):
    with tf.variable_scope('RNN_{}'.format(i)):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        in_cells.append(cell)
in_cell = tf.nn.rnn_cell.MultiRNNCell(in_cells)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(in_cell, enc_inp, sequence_length=output_lengths, dtype=tf.float32)

out_cells = []
for i in range(layers_stacked_count):
    with tf.variable_scope('RNN_{}'.format(i)):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        out_cells.append(cell)
out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=hidden_dim, memory=encoder_outputs)
out_cell = tf.contrib.seq2seq.AttentionWrapper(out_cell, attention_mechanism, alignment_history=True)
out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, 2)

# helper = tf.contrib.seq2seq.TrainingHelper(expected_sparse_output, output_lengths)
# decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper, initial_state=out_cell.zero_state(batch_size, tf.float32))
# outputs_with_helper = tf.contrib.seq2seq.dynamic_decode(decoder)

initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
initial_state = initial_state.clone(cell_state=encoder_final_state)

outputs = []
for i in range(steps_out):
    if i == 0:
        # output, state = out_cell(tf.zeros((batch_size, 2)), initial_state)
        output, state = out_cell(expected_sparse_output[:, i, :], initial_state)
        print(state)
        # tmp = out_cell(expected_sparse_output[:, i, :], initial_state)
    else:
        output, state = out_cell(expected_sparse_output[:, i, :], state)
    outputs.append(output)

outputs = tf.convert_to_tensor(outputs)
reshaped_outputs = []
for i in range(batch_size):
    reshaped_outputs.append(outputs[:, i, :])
outputs = tf.convert_to_tensor(reshaped_outputs)

# cell_outs = tf.convert_to_tensor(cell_outs)
# reshaped_cell_outs = []
# for i in range(batch_size):
#     reshaped_cell_outs.append(cell_outs[:, i, :])
# cell_outs = tf.convert_to_tensor(reshaped_cell_outs)

predict_outputs = []
for i in range(steps_out):
    if i == 0:
        output, state = out_cell(tf.zeros((batch_size, 2)), initial_state)
    else:
        output, state = out_cell(output, state)
    predict_outputs.append(output)

predict_outputs = tf.convert_to_tensor(predict_outputs)
predict_reshaped_outputs = []
for i in range(batch_size):
    predict_reshaped_outputs.append(predict_outputs[:, i, :])
predict_outputs = tf.convert_to_tensor(predict_reshaped_outputs)

learning_rate = 0.007  # Small lr helps not to diverge during training.
nb_iters = 150  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0  # L2 regularization of weights - avoids overfitting

reg_loss_1 = 0
for tf_var in tf.trainable_variables():
    reg_loss_1 += tf.reduce_mean(tf.nn.l2_loss(tf_var))
loss_1 = tf.reduce_mean(tf.nn.l2_loss(predict_outputs - expected_sparse_output)) + lambda_l2_reg*reg_loss_1
train_1 = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum).minimize(loss_1)

reg_loss_0 = 0
for tf_var in tf.trainable_variables():
    reg_loss_0 += tf.reduce_mean(tf.nn.l2_loss(tf_var))
loss_0 = tf.reduce_mean(tf.nn.l2_loss(outputs - expected_sparse_output)) + lambda_l2_reg*reg_loss_0
train_0 = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum).minimize(loss_0)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    x_batch, y_batch = ts_data.next_batch(batch_size, steps_in, steps_out)
    feed_dict = {enc_inp: x_batch, expected_sparse_output: y_batch, keep_prob: 1}
    for i in range(1):
        x_batch, y_batch = ts_data.next_batch(batch_size, steps_in, steps_out)
        feed_dict = {enc_inp: x_batch, expected_sparse_output: y_batch, keep_prob: 1}
        print(sess.run(state.alignments, feed_dict=feed_dict))
        # print(sess.run(outputs_with_helper, feed_dict=feed_dict))
        if i % 2 == 0:
            sess.run(train_0, feed_dict=feed_dict)
        else:
            sess.run(train_1, feed_dict=feed_dict)
        if i % 100 == 0:
            print(sess.run(loss_1, feed_dict=feed_dict))

    x, y = ts_data.next_batch(batch_size, steps_in, steps_out)
    feed_dict = {enc_inp: x, expected_sparse_output: y, keep_prob: 1}
    outputs = sess.run(predict_outputs, feed_dict=feed_dict)

# for j in range(batch_size):
#     plt.figure(figsize=(12, 3))

#     for k in range(output_dim):
#         past = x[j, :, k]
#         expected = y[j, :, k]
#         pred = outputs[j, :, k]

#         label1 = "Seen (past) values" if k == 0 else "_nolegend_"
#         label2 = "True future values" if k == 0 else "_nolegend_"
#         label3 = "Predictions" if k == 0 else "_nolegend_"
#         plt.plot(range(len(past)), past, "o--b", label=label1)
#         plt.plot(range(len(past), len(expected)+len(past)),
#                  expected, "x--b", label=label2)
#         plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y", label=label3)

#     plt.legend(loc='best')
#     plt.title("Predictions v.s. true values")
#     plt.show()
