import numpy as np
import tensorflow as tf
import time
import itertools

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name

        self.learning_rate = Config.LEARNING_RATE

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()
                # to make it run on a few threads
                # config = tf.ConfigProto(
                #         intra_op_parallelism_threads=4,
                #         inter_op_parallelism_threads=4
                # )
                config = tf.ConfigProto()
                config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
                self.sess = tf.Session(
                    graph=self.graph,
                    config=config
                )
                self._create_tensor_board()
                self.saver = tf.train.Saver()
                self.sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint("./checkpoint/")
                if latest_checkpoint and Config.RESTORE:
                    print("Restoring Parameters from latest checkpoint:")
                    self.saver.restore(self.sess, latest_checkpoint)

    def _create_graph(self):
        self.state = tf.placeholder(tf.float32, shape=[None, Config.NUM_OF_CUSTOMERS+1, 2], name='State')
        self.sampled_cost = tf.placeholder(tf.float32, [None, 1], name='Sampled_Cost')
        self.batch_size = tf.shape(self.state)[0]
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.input_lengths = tf.convert_to_tensor([Config.NUM_OF_CUSTOMERS]*(self.batch_size))
        tf.summary.scalar("batch_size", self.batch_size)

        # tmp for random action
        self.all_actions = []
        for i in itertools.permutations(range(1, Config.NUM_OF_CUSTOMERS), Config.NUM_OF_CUSTOMERS-1):
            route = np.zeros(Config.NUM_OF_CUSTOMERS+1, dtype=np.int32)
            for j in range(len(i)):
                route[j+1] = i[j]
            self.all_actions.append(route)
        self.all_actions = tf.convert_to_tensor(np.asarray(self.all_actions))

        self.action = tf.gather(self.all_actions, tf.random_uniform(
            dtype=tf.int32, minval=0, maxval=tf.shape(self.all_actions)[0], shape=[self.batch_size]))

        self.embeded_enc_inputs = tf.layers.conv1d(self.state, 1, 1)
        self.in_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                self.cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
                self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                self.in_cells.append(self.cell)
        self.in_cell = tf.nn.rnn_cell.MultiRNNCell(self.in_cells)
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.in_cell,
                                                                           self.embeded_enc_inputs,
                                                                           dtype=tf.float32)
        self.dense_layer_input = tf.layers.flatten(tf.transpose(tf.convert_to_tensor(self.encoder_final_state), perm=[1, 0, 2]))
        self.dense_layer = tf.layers.dense(
            self.dense_layer_input,
            Config.DNN_HIDDEN_DIM,
            activation=tf.nn.relu)
        with tf.name_scope("base_line"):
            self.base_line_est = tf.layers.dense(self.dense_layer, 1, activation=None)
        tf.summary.scalar("base_line_est", tf.reduce_mean(self.base_line_est))

        with tf.name_scope("loss"):
            self.base_line_loss = tf.reduce_mean(self.base_line_est - self.sampled_cost)
        tf.summary.scalar("loss", self.base_line_loss)

        with tf.name_scope("train"):
            self.base_line_train = tf.train.AdamOptimizer(Config.LEARNING_RATE).\
                                   minimize(self.base_line_loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state):
        prediction = self.sess.run([self.action, self.base_line_est], feed_dict={self.state: state, self.keep_prob: .5})
        return prediction

    def train(self, state, action, sampled_cost, trainer_id):
        step = self.get_global_step()
        feed_dict = {self.state: state, self.sampled_cost: sampled_cost, self.keep_prob: .5}
        summary, _ = self.sess.run([self.merged, self.base_line_train], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)
        if step % 1000 == 0:
            self.saver.save(self.sess, "./checkpoint/model.ckpt")

    def _create_tensor_board(self):
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name + '%f' % time.time())
        self.log_writer.add_graph(self.sess.graph)
