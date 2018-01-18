import numpy as np
import tensorflow as tf
import time
import itertools

from Config import Config

# BUGS
# there is something wrong with the number of customers to number of actions. random actions goes up to 7 and it should go up to 8

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
        self.current_location = tf.placeholder(tf.float32, shape=[None, 2], name='Current_Location')
        self.sampled_cost = tf.placeholder(tf.float32, [None, 1], name='Sampled_Cost')
        self.batch_size = tf.shape(self.state)[0]
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.input_lengths = tf.convert_to_tensor([Config.NUM_OF_CUSTOMERS]*(self.batch_size))
        self.or_action = tf.placeholder(tf.int32, shape=[None, Config.NUM_OF_CUSTOMERS+1])
        self.or_cost = tf.placeholder(tf.float32, shape=[None, 1])
        tf.summary.scalar("batch_size", self.batch_size)

        # tmp for random action
        self.all_actions = []
        for i in itertools.permutations(range(0, Config.NUM_OF_CUSTOMERS+1), Config.NUM_OF_CUSTOMERS+1):
            route = np.zeros(Config.NUM_OF_CUSTOMERS+1, dtype=np.int32)
            for j in range(len(i)):
                route[j] = i[j]
            self.all_actions.append(route)
        self.all_actions = tf.convert_to_tensor(np.asarray(self.all_actions))

        self.action = tf.gather(self.all_actions, tf.random_uniform(
            dtype=tf.int32, minval=0, maxval=tf.shape(self.all_actions)[0], shape=[self.batch_size]))

        self.enc_inputs = self.state
        self.embeded_enc_inputs = tf.layers.conv1d(self.state, Config.RNN_HIDDEN_DIM, kernel_size=1, use_bias=False)

        # Above should be the same as below
        # input_embed = tf.get_variable(
        #     "input_embed", [1, 2, Config.RNN_HIDDEN_DIM],
        #     initializer=tf.random_uniform_initializer(-1, 1))
        # self.embeded_enc_inputs = tf.nn.conv1d(
        #     self.enc_inputs, input_embed, 1, "VALID")
        # self.batch_size_embed = tf.concat([tf.shape(self.embeded_enc_inputs), tf.shape(self.state)], axis=0)

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

        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                        memory=self.encoder_outputs)

        self.out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                self.cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
                self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                self.out_cells.append(self.cell)
        self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
        # not sure about alignment_history
        self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism, alignment_history=True)
        self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)  # not sure why +2

        self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
        self.initial_state = self.initial_state.clone(cell_state=self.encoder_final_state)

        # Determine Action
        # Make sure paramaters for decoder are trainable
        # I could do a logit of the output of the decoder like traditional seq2seq
        self.outputs = []
        self.locations = []
        for i in range(9):
            if i == 0:
                output, rnn_state = self.out_cell(self.current_location, self.initial_state)
                self.logits = tf.expand_dims(tf.nn.softmax(output), 1)
            else:
                output, rnn_state = self.out_cell(self.next_location, rnn_state)
                self.logits = tf.concat([self.logits, tf.expand_dims(tf.nn.softmax(output), 1)], 1)
            if i == 0:
                self.alignments = tf.argmax(tf.nn.softmax(output), axis=1, output_type=tf.int32)
                self.all_alignments = tf.reshape(self.alignments, [-1, 1])
            else:
                self.mask = tf.reduce_sum(
                    tf.one_hot(self.all_alignments, depth=tf.shape(self.state)[1], on_value=1.0, off_value=0.0, dtype=tf.float32),
                    axis=1)
                # self.rnn_state = rnn_state.alignments
                self.alignments = tf.argmax(rnn_state.alignments - self.mask, axis=1, output_type=tf.int32)
                self.all_alignments = tf.concat([self.all_alignments, tf.reshape(self.alignments, [-1, 1])], axis=1)
            self.next_location = tf.gather_nd(
                self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                       tf.reshape(tf.argmax(tf.nn.softmax(output), axis=1, output_type=tf.int32), [-1, 1])], 1)
            )
            self.locations.append(self.next_location)
            self.outputs.append(output)
            # self.step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=self.or_action[:, i], logits=output)

        self.new_action = self.all_alignments
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits,
            targets=self.or_action,
            weights=tf.ones([self.batch_size, Config.NUM_OF_CUSTOMERS+1]))
        tf.summary.scalar("loss", self.loss)

        # tf.reshape(tf.reshape(tf.tile(tf.expand_dims([1, 2, 3, 4], -1),  [1, 2]), [-1])[1:-1], [-1, 2])
        # tf.reshape(tf.reshape(
        #     tf.tile(tf.expand_dims(self.new_action, -1), [1, 1, 2]),
        #     [self.batch_size, -1])[:, 1:-1], [self.batch_size, -1, 2])
        # loss = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(tf.cast(tf.not_equal(tmp, tmp1), tf.int32), axis=1), 0), tf.int32))

        # tf.sets.set_difference()  # turns it into a weird list
        # tf.setdiff1d()            # may require it to be 1d

        # self.dense_layer_input = tf.layers.flatten(tf.transpose(tf.convert_to_tensor(self.encoder_final_state), perm=[1, 0, 2]))
        # self.dense_layer = tf.layers.dense(
        #     self.dense_layer_input,
        #     Config.DNN_HIDDEN_DIM,
        #     activation=tf.nn.relu)
        with tf.name_scope("base_line"):
            # self.base_line_est = tf.layers.dense(self.dense_layer, 1, activation=None)
            self.base_line_est = tf.zeros(shape=[self.batch_size, 1])
        # tf.summary.scalar("base_line_est", tf.reduce_mean(self.base_line_est))

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(Config.LEARNING_RATE).\
                                   minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state, current_locations):
        feed_dict = {self.state: state, self.current_location: current_locations, self.keep_prob: .5}
        prediction = self.sess.run([self.action, self.base_line_est], feed_dict=feed_dict)
        return prediction

    def train(self, state, current_location, action, or_action, sampled_cost, or_cost, trainer_id):
        step = self.get_global_step()
        feed_dict = {self.state: state, self.current_location: current_location, self.or_action: or_action,
                     self.sampled_cost: sampled_cost, self.or_cost: or_cost, self.keep_prob: .5}
        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)
        if step % 100 == 0:
            print(self.sess.run(self.loss, feed_dict=feed_dict))
        if step % 1000 == 0:
            self.saver.save(self.sess, "./checkpoint/model.ckpt")

    def _create_tensor_board(self):
        name = "logs/%s" % self.model_name + '%f' % time.time()
        print("Running Model ", name)
        self.log_writer = tf.summary.FileWriter(name)
        self.log_writer.add_graph(self.sess.graph)
