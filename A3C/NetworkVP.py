import numpy as np
import tensorflow as tf
import time
import itertools

from Config import Config


# I need to make sure that the data input doesnt need to always have (0, 0) as first input for or_action
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
                config = tf.ConfigProto(
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4
                )
                # config = tf.ConfigProto()
                # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
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
        # self.mask = tf.placeholder(tf.float32, shape=[None, Config.NUM_OF_CUSTOMERS+1])
        tf.summary.scalar("batch_size", self.batch_size)
        tf.summary.scalar("difference_in_length", tf.reduce_mean(self.sampled_cost - self.or_cost))

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

        # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        self.in_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                self.cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
                # self.cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(Config.RNN_HIDDEN_DIM)
                # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                self.in_cells.append(self.cell)
        self.in_cell = tf.nn.rnn_cell.MultiRNNCell(self.in_cells)
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.in_cell,
                                                                           self.enc_inputs,
                                                                           dtype=tf.float32)

        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                        memory=self.encoder_outputs)

        # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        self.out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                self.cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
                # self.cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(Config.RNN_HIDDEN_DIM)
                # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                self.out_cells.append(self.cell)
        self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
        self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism)
        self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)

        self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
        self.initial_state = self.initial_state.clone(cell_state=self.encoder_final_state)

        # Determine Action
        # Make sure paramaters for decoder are trainable
        # I could do a logit of the output of the decoder like traditional seq2seq

        # batch_size = self.batch_size

        # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     embedding=self.state,
        #     start_tokens=tf.tile([0], [batch_size]),
        #     end_token=9)

        # with tf.variable_scope("mask"):
        #     self.mask = tf.get_variable("m", [tf.shape(self.state)[0], tf.shape(self.state)[1]], dtype=tf.float32, trainable=False,
        #                                 initializer=tf.zeros)

        # self.mask = tf.Variable(
        #     [],
        #     dtype=tf.float32, validate_shape=False)

        # self.mask = tf.assign(self.mask, tf.zeros([self.batch_size, Config.NUM_OF_CUSTOMERS+1]), validate_shape=False)

        self.mask = tf.Variable(tf.zeros([Config.MAX_BATCH_SIZE, Config.NUM_OF_CUSTOMERS+1]), dtype=tf.float32, trainable=False)

        # self.tmp = tf.get_variable("tmp", [self.state.get_shape().as_list()[0], Config.NUM_OF_CUSTOMERS+1], initializer=tf.zeros_initializer())
        # self.mask = tf.zeros([self.batch_size, tf.shape(self.state)[1]])

        # self.all_alignments = tf.Print(self.all_alignments, [self.all_alignments, self.all_alignments.read_value()])

        def initialize_fn():
            return (tf.tile([False], [self.batch_size]), tf.zeros([self.batch_size, 2]))

        def sample_fn(time, outputs, state):
            # mask = tf.cond(tf.shape(self.all_alignments)[0] > 1,
            #                lambda: tf.reduce_sum(
            #         tf.one_hot(self.all_alignments, depth=tf.shape(self.state)[1], on_value=1.0, off_value=0.0, dtype=tf.float32),
            #         axis=1),
            #                lambda: tf.zeros(tf.shape(outputs)))
            # mask = tf.cond(time > 0,
            #                lambda: self.all_alignments,
            #                lambda: tf.zeros(tf.shape(outputs)))
            # mask = tf.Print(mask, [mask, tf.shape(mask)])
            # outputs = tf.Print(outputs, [outputs, tf.shape(outputs)])
            # self.all_alignments = tf.Print(self.all_alignments, [self.all_alignments, self.all_alignments.read_value()])
            # with tf.control_dependencies([self.mask]):
            # with tf.variable_scope("mask"):
            #     sample_ids = tf.stop_gradient(tf.argmax(tf.nn.softmax(outputs) - tf.get_variable("m", [self.batch_size, Config.NUM_OF_CUSTOMERS+1]), axis=1, output_type=tf.int32))

            mask_subset = tf.gather(self.mask, tf.range(0, tf.shape(outputs)[0]))
            sample_ids = tf.argmax(tf.nn.softmax(outputs)-mask_subset, axis=1, output_type=tf.int32)
            # self.all_alignments = self.all_alignments + tf.cast(tf.one_hot(sample_ids, depth=9, on_value=1), dtype=tf.float32)
            # assign_op = tf.assign(self.mask, tf.ones(tf.shape(self.mask)), name="control_dependencies_thing")
            # assign = tf.assign(self.all_alignments,
            #                    tf.concat([self.all_alignments, tf.reshape(sample_ids, [-1, 1])], axis=1),
            #                    validate_shape=False)
            # # # assign = tf.Print(self.all_alignments, [self.all_alignments])
            # with tf.control_dependencies([sample_ids]):
                # assign_op = tf.stop_gradient(tf.cond(
                #     tf.equal(time, 0),
                #     lambda: tf.assign(self.mask, tf.one_hot(
                #         sample_ids,
                #         depth=tf.shape(self.state)[1],
                #         on_value=1.0,
                #         off_value=0.0,
                #         dtype=tf.float32)),
                #     lambda: tf.assign(self.mask, self.mask + tf.one_hot(
                #         sample_ids,
                #         depth=tf.shape(self.state)[1],
                #         on_value=1.0,
                #         off_value=0.0,
                #         dtype=tf.float32))
                #     ))

            # assign_op = tf.assign(self.mask, self.mask + tf.one_hot(
            #         sample_ids,
            #         depth=tf.shape(self.state)[1],
            #         on_value=1.0,
            #         off_value=0.0,
            #         dtype=tf.float32), validate_shape=False)

            # tmp1 = tf.one_hot(sample_ids, depth=tf.shape(self.state)[1], on_value=1.0, off_value=0.0, dtype=tf.float32)
            # tmp1 = tf.Print(tmp1, [tmp1], summarize=100)
            # tmp = self.mask + tf.concat(
            #                         [tmp1,
            #                          tf.zeros([10-self.batch_size, tf.shape(self.state)[1]])],
            #                         axis=0)
            # tmp = tf.Print(tmp, [tmp], summarize=100)
            assign_op = tf.cond(tf.equal(time, 0),
                                lambda: tf.assign(self.mask, tf.zeros_like(self.mask) + tf.concat(
                                    [tf.one_hot(sample_ids, depth=tf.shape(self.state)[1], dtype=tf.float32),
                                     tf.zeros([10-self.batch_size, tf.shape(self.state)[1]])],
                                    axis=0)),
                                lambda: tf.assign(self.mask, self.mask + tf.concat(
                                    [tf.one_hot(sample_ids, depth=tf.shape(self.state)[1], dtype=tf.float32),
                                     tf.zeros([10-self.batch_size, tf.shape(self.state)[1]])],
                                    axis=0)))

            # self.mask = tf.add(self.mask, tf.one_hot(
            #         sample_ids,
            #         depth=tf.shape(self.state)[1],
            #         on_value=1.0,
            #         off_value=0.0,
            #         dtype=tf.float32))
            # self.mask = tf.Print(self.mask, [self.mask], summarize=30)
            with tf.control_dependencies([assign_op]):
                return tf.identity(sample_ids)

        def next_inputs_fn(time, outputs, state, sample_ids):
            finished = tf.tile([tf.equal(time, 8)], [self.batch_size])
            next_inputs = tf.stop_gradient(tf.gather_nd(
                self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                       tf.reshape(sample_ids, [-1, 1])], 1)
            ))
            next_state = state
            return (finished, next_inputs, next_state)

        helper = tf.contrib.seq2seq.CustomHelper(
            initialize_fn=initialize_fn,
            sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn
        )

        decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, helper, self.initial_state)

        self.final_output, self.final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.final_action = self.final_output.sample_id

        # self.all_alignments = tf.Print(self.all_alignments, [self.all_alignments, self.all_alignments.read_value()])

        # def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        #     cell_output = self.out_cell()
        #     if cell_state is None:
        #         cell_state = enc_final_states
        #         next_input = cell_input
        #         done = tf.zeros([batch_size,], dtype=tf.bool)
        #     else:
        #         sampled_idx = helper.sample(time, outputs, state)
        #         next_input = helper.next_inputs(time, cell_output, state, sample_ids)
        #         done = tf.equal()
        #     done = tf.cond(tf.greater(time, maximum_length),
        #                    lambda: tf.ones([batch_size, ], dtype=tf.bool),
        #                    lambda: done)
        #     return (done, cell_state, next_input, cell_output, context_state)

        # self.outputs = []
        # self.locations = []
        # for i in range(9):
        #     if i == 0:
        #         output, rnn_state = self.out_cell(self.current_location, self.initial_state)
        #         self.logits = tf.expand_dims(tf.nn.softmax(output), 1)
        #     else:
        #         output, rnn_state = self.out_cell(self.next_location, rnn_state)
        #         self.logits = tf.concat([self.logits, tf.expand_dims(tf.nn.softmax(output), 1)], 1)
        #     if i == 0:
        #         self.alignments = tf.argmax(tf.nn.softmax(output), axis=1, output_type=tf.int32)
        #         self.all_alignments = tf.reshape(self.alignments, [-1, 1])
        #     else:
        #         self.mask = tf.reduce_sum(
        #             tf.one_hot(self.all_alignments, depth=tf.shape(self.state)[1], on_value=1.0, off_value=0.0, dtype=tf.float32),
        #             axis=1)
        #         # self.rnn_state = rnn_state.alignments
        #         self.alignments = tf.argmax(rnn_state.alignments - self.mask, axis=1, output_type=tf.int32)
        #         self.all_alignments = tf.concat([self.all_alignments, tf.reshape(self.alignments, [-1, 1])], axis=1)
        #     self.next_location = tf.gather_nd(
        #         self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
        #                                tf.reshape(tf.argmax(tf.nn.softmax(output), axis=1, output_type=tf.int32), [-1, 1])], 1)
        #     )
        #     self.locations.append(self.next_location)
        #     self.outputs.append(output)
        #     # self.step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     #     labels=self.or_action[:, i], logits=output)
        # self.new_action = self.all_alignments

        # self.weights = tf.concat([tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, Config.NUM_OF_CUSTOMERS])], axis=1)
        self.weights = tf.ones([self.batch_size, 9])

        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.final_output.rnn_output,
            # logits=self.logits,
            targets=self.or_action,
            weights=self.weights,
            softmax_loss_function=tf.losses.sparse_softmax_cross_entropy
        )
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

        # self.tmp = tf.one_hot(tf.ones([4], dtype=tf.int32), depth=9)
        # self.test = tf.concat([self.tmp, tf.zeros([10-tf.shape(self.tmp)[0], 9])], axis=0)

        with tf.name_scope("train"):
            self.lr = tf.train.exponential_decay(
                Config.LEARNING_RATE, self.global_step, 5000,
                .96, staircase=True, name="learning_rate")
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state, current_location):
        # feed_dict = {self.state: state, self.current_location: current_location, self.keep_prob: .5, self.or_action: np.zeros((state.shape[0], 9), dtype=np.int32), self.sampled_cost: np.zeros((state.shape[0], 1)), self.or_cost: np.zeros((state.shape[0], 1))}
        # state = np.zeros((4, 9, 2))
        # current_location = np.zeros((4, 2))
        # feed_dict = {self.state: state, self.current_location: current_location, self.keep_prob: .5, self.or_action: np.zeros((state.shape[0], 9), dtype=np.int32), self.sampled_cost: np.zeros((state.shape[0], 1)), self.or_cost: np.zeros((state.shape[0], 1))}
        feed_dict = {self.state: state, self.current_location: current_location, self.keep_prob: .5}
        # , self.mask: np.zeros((state.shape[0], state.shape[1]))}
        # self.sess.run([self.mask_init_op], feed_dict=feed_dict)
        # print(self.sess.run([self.final_action], feed_dict=feed_dict))
        # I would like to name action for first imput here but gives odd error when i do
        # step = self.get_global_step()
        # summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        prediction = self.sess.run([self.final_action, self.base_line_est], feed_dict=feed_dict)
        # print(self.sess.run(self.final_action, feed_dict=feed_dict))
        # print(out)
        # print(out)
        # print(self.sess.run([self.action, self.base_line_est, self.state, self.current_location], feed_dict=feed_dict))
        return prediction

    def train(self, state, current_location, action, or_action, sampled_cost, or_cost, trainer_id):
        # print(state.shape, type(state))
        # print(current_location.shape, type(current_location))
        step = self.get_global_step()
        feed_dict = {self.state: state, self.current_location: current_location, self.or_action: or_action,
                     self.sampled_cost: sampled_cost, self.or_cost: or_cost, self.keep_prob: .5}
        # print(self.sess.run(self.final_action, feed_dict=feed_dict))

        # test = tf.assign(self.mask, tf.concat())
        # print("test")
        # print(self.sess.run(self.test, feed_dict=feed_dict))
        # state = np.zeros((3, 9, 2))
        # current_location = np.zeros((3, 2))
        # feed_dict = {self.state: state, self.current_location: current_location, self.keep_prob: .5, self.or_action: np.zeros((state.shape[0], 9), dtype=np.int32), self.sampled_cost: np.zeros((state.shape[0], 1)), self.or_cost: np.zeros((state.shape[0], 1))}
        # , self.mask: np.zeros((state.shape[0], state.shape[1]))}
        # self.sess.run([self.state, self.or_action, self.or_cost, self.sampled_cost], feed_dict=feed_dict)
        # print(type(self.sess.run(self.action, feed_dict=feed_dict)))
        # print(self.sess.run(self.final_action, feed_dict=feed_dict))
        # print(self.sess.run(tf.argmax(self.final_state.alignment_history.stack()), feed_dict=feed_dict))
        # print(self.sess.run(tf.one_hot(self.all_alignments, depth=tf.shape(self.state)[1], on_value=1.0, off_value=0.0, dtype=tf.float32), feed_dict=feed_dict))

        # state = np.zeros((5, 9, 2))
        # current_location = np.zeros((5, 2))
        # feed_dict = {self.state: state, self.current_location: current_location, self.or_action: np.zeros((state.shape[0], 9), dtype=np.int32), self.sampled_cost: np.zeros((state.shape[0], 1)), self.or_cost: np.zeros((state.shape[0], 1)), self.keep_prob: .5}
        # print(self.sess.run(self.final_action, feed_dict=feed_dict))

        # print("running")

        # print(self.sess.run(self.tmp, feed_dict=feed_dict))
        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        if step % 100 == 0:
            self.log_writer.add_summary(summary, step)
            print(self.sess.run(self.loss, feed_dict=feed_dict))
        if step % 1000 == 0:
            self.saver.save(self.sess, "./checkpoint/model.ckpt")

            # print(self.sess.run(self.loss, feed_dict=feed_dict))
            # print(self.sess.run([tf.reduce_mean(self.sampled_cost - self.or_cost)], feed_dict=feed_dict))
            # print(self.sess.run([self.final_output.sample_id, self.action], feed_dict=feed_dict))
            # print(self.sess.run(, feed_dict=feed_dict))
            # print()
            # combine them so test pred are next to eachother
            # print(self.sess.run([self.or_action, self.new_action], feed_dict=feed_dict))

    def _create_tensor_board(self):
        name = "logs/%s" % self.model_name + '%f' % time.time()
        print("Running Model ", name)
        self.log_writer = tf.summary.FileWriter(name)
        self.log_writer.add_graph(self.sess.graph)
