import tensorflow as tf
import time
import numpy as np
from MaskWrapper import MaskWrapper

from Config import Config

from tensorflow.python.layers.core import Dense


# I need to make sure that the data input doesnt need to always have (0, 0) as first input for or_action
# add error message when final_action has duplicate entries
class NetworkVP:
    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name
        # self.learning_rate = Config.LEARNING_RATE

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            # with tf.device(self.device):
            self._create_graph()
            # to make it run on a few threads
            config = tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1
            )
            # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            # config = tf.ConfigProto()
            # log_device_placement=True
            # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            self.sess = tf.Session(
                graph=self.graph,
                config=config
            )
            # self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            if Config.TRAIN:
                self._create_tensor_board()
            # latest_checkpoint = tf.train.latest_checkpoint("./checkpoint/")
            # if latest_checkpoint and Config.RESTORE:
            #     print("Restoring Parameters from latest checkpoint:")
            #     self.saver.restore(self.sess, latest_checkpoint)

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
        self.difference_in_length = tf.reduce_mean(self.sampled_cost - self.or_cost)
        self.start_tokens = tf.zeros([self.batch_size], tf.int32)
        self.end_token = -1
        tf.summary.scalar("batch_size", self.batch_size)
        tf.summary.scalar("difference_in_length", self.difference_in_length)
        tf.summary.scalar("Config.LAYERS_STACKED_COUNT", Config.LAYERS_STACKED_COUNT)
        tf.summary.scalar("RNN_HIDDEN_DIM", Config.RNN_HIDDEN_DIM)
        tf.summary.scalar("RUN_TIME", Config.RUN_TIME)

        self.enc_inputs = self.state

        def _build_rnn_cell():
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
            if Config.CELL_TYPE == 0:
                return tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)

        if Config.DIRECTION == 1:
            self.in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.in_cells.append(self.cell)
            self.in_cell = tf.nn.rnn_cell.MultiRNNCell(self.in_cells)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.in_cell,
                                                                         self.enc_inputs,
                                                                         dtype=tf.float32)
        if Config.DIRECTION == 2:
            self.in_cell = tf.contrib.rnn.BasicLSTMCell(Config.RNN_HIDDEN_DIM)
            (bi_outputs, (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                             cell_fw=self.in_cell, cell_bw=self.in_cell, inputs=self.enc_inputs, dtype=tf.float32)
            encoder_state_c = tf.concat(
                (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
            encoder_state_h = tf.concat(
                (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
            self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            self.encoder_outputs = tf.concat(bi_outputs, -1)
        if Config.DIRECTION == 3:
            self.in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.in_cells.append(self.cell)
            (self.encoder_outputs, self.encoder_fw_state, self.encoder_bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                             cells_fw=self.in_cells, cells_bw=self.in_cells, inputs=self.enc_inputs, dtype=tf.float32)
            # self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            self.encoder_state = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                # state_slice_shape = [1, tf.shape(self.encoder_fw_state[0])[1], tf.shape(self.encoder_fw_state[0])[2]]
                # encoder_fw_state_c = tf.slice(self.encoder_fw_state[i], [0, 0, 0], state_slice_shape)
                # encoder_bw_state_c = tf.slice(self.encoder_bw_state[i], [0, 0, 0], state_slice_shape)
                # encoder_fw_state_h = tf.slice(self.encoder_fw_state[i], [1, 0, 0], state_slice_shape)
                # encoder_bw_state_h = tf.slice(self.encoder_bw_state[i], [1, 0, 0], state_slice_shape)
                encoder_state_c = tf.concat(
                    (self.encoder_fw_state[i].c, self.encoder_bw_state[i].c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (self.encoder_fw_state[i].h, self.encoder_bw_state[i].h), 1, name='bidirectional_concat_h')
                self.encoder_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
            # self.encoder_state = tf.while_loop(lambda i: i < Config.LAYERS_STACKED_COUNT, body, tf.constant([0]))
            self.encoder_state = tuple(self.encoder_state)

            # use tf.slice to get column of .c and column of .h then concat
            # encoder_fw_state = tf.convert_to_tensor(encoder_fw_state)
            # encoder_bw_state = tf.convert_to_tensor(encoder_bw_state)
            # encoder_fw_state = tf.Print(encoder_fw_state, [tf.shape(encoder_fw_state)], summarize=100)
            # with tf.control_dependencies([encoder_fw_state]):
            #     self.encoder_outputs = tf.identity(self.encoder_outputs)
            # state_slice_shape = [tf.shape(encoder_fw_state)[0], 1, tf.shape(encoder_fw_state)[2], tf.shape(encoder_fw_state)[3]]
            # encoder_fw_state_c = tf.slice(encoder_fw_state, [0, 0, 0, 0], state_slice_shape)
            # encoder_bw_state_c = tf.slice(encoder_bw_state, [0, 0, 0, 0], state_slice_shape)
            # encoder_fw_state_h = tf.slice(encoder_fw_state, [0, 1, 0, 0], state_slice_shape)
            # encoder_bw_state_h = tf.slice(encoder_bw_state, [0, 1, 0, 0], state_slice_shape)
            # encoder_state_c = tf.concat(
            #     (encoder_fw_state_c, encoder_bw_state_c), 1, name='bidirectional_concat_c')
            # encoder_state_h = tf.concat(
            #     (encoder_fw_state_h, encoder_bw_state_h), 1, name='bidirectional_concat_h')
            # encoder_state_c = tf.Print(encoder_state_c, [tf.shape(encoder_state_c)])
            # encoder_state_h = tf.Print(encoder_state_h, [tf.shape(encoder_state_h)])
            # self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            # self.encoder_state = tf.Print(self.encoder_state, [tf.shape(self.encoder_state)])

        if Config.DIRECTION == 4:
            self.in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.in_cells.append(self.cell)
            self.in_cell = tf.nn.rnn_cell.MultiRNNCell(self.in_cells)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.in_cell,
                                                                         self.enc_inputs,
                                                                         dtype=tf.float32)

        # def initialize_fn():
        #     return (tf.tile([False], [self.batch_size]), tf.zeros([self.batch_size, 2]))

        # # # self.mask = tf.contrib.framework.local_variable(tf.zeros([Config.MAX_BATCH_SIZE, Config.NUM_OF_CUSTOMERS+1]))
        # # # self.mask = tf.get_variable("self.mask", shape=[Config.MAX_BATCH_SIZE, Config.NUM_OF_CUSTOMERS+1],
        # # #                             initializer=tf.zeros_initializer, trainable=False)
        # #                             # collections=[tf.GraphKeys.LOCAL_VARIABLES],
        # # # self.mask = tf.zeros([Config.MAX_BATCH_SIZE, Config.NUM_OF_CUSTOMERS+1])

        # def sample_fn(time, outputs, state):
        #     # mask_subset = tf.gather(self.mask, tf.range(0, tf.shape(outputs)[0]))
        #     sample_ids = tf.argmax(tf.nn.softmax(outputs), axis=1, output_type=tf.int32)
        #     # assign_op = tf.cond(tf.equal(time, 0),
        #     #                     lambda: tf.assign(self.mask, tf.zeros_like(self.mask) + tf.concat(
        #     #                         [tf.one_hot(sample_ids, depth=tf.shape(outputs)[1], dtype=tf.float32),
        #     #                          tf.zeros([Config.MAX_BATCH_SIZE-tf.shape(outputs)[0], tf.shape(self.state)[1]])],
        #     #                         axis=0)),
        #     #                     lambda: tf.assign(self.mask, self.mask + tf.concat(
        #     #                         [tf.one_hot(sample_ids, depth=tf.shape(outputs)[1], dtype=tf.float32),
        #     #                          tf.zeros([Config.MAX_BATCH_SIZE-self.batch_size, tf.shape(self.state)[1]])],
        #     #                         axis=0)))
        #     # with tf.control_dependencies([assign_op]):
        #     return tf.identity(sample_ids)

        # def next_inputs_fn(time, outputs, state, sample_ids):
        #     finished = tf.tile([tf.equal(time, Config.NUM_OF_CUSTOMERS)], [self.batch_size])
        #     next_inputs = tf.stop_gradient(tf.gather_nd(
        #         self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
        #                                tf.reshape(sample_ids, [-1, 1])], 1)
        #     ))
        #     next_state = state
        #     return (finished, next_inputs, next_state)

        # helper = tf.contrib.seq2seq.CustomHelper(
        #     initialize_fn=initialize_fn,
        #     sample_fn=sample_fn,
        #     next_inputs_fn=next_inputs_fn
        # )

        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            lambda sample_ids: tf.gather_nd(
                self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                       tf.reshape(sample_ids, [-1, 1])], 1)
            ),
            # lambda sample_ids: tf.zeros([tf.shape(sample_ids)[0], 2]),
            self.start_tokens,
            self.end_token)

        self.training_inputs = tf.expand_dims(tf.gather_nd(self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                                                                  tf.reshape(self.or_action[:, 0], [-1, 1])], 1)), 1)
        for i in range(1, Config.NUM_OF_CUSTOMERS+1):
            self.training_inputs = tf.concat(
                [self.training_inputs, tf.expand_dims(
                    tf.gather_nd(self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                                        tf.reshape(self.or_action[:, i], [-1, 1])], 1)), 1)],
                1)
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.training_inputs, tf.fill([self.batch_size], Config.NUM_OF_CUSTOMERS+1))

        if Config.DIRECTION == 1:
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                            memory=self.encoder_outputs)
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
            # last_state = tf.concat([tf.zeros([self.batch_size, 8]), tf.ones([self.batch_size, 1])], 1)
            # self.tmp0 = helper.sample(0, last_state, self.initial_state)
            # self.tmp1 = helper.next_inputs(0, last_state, self.initial_state, self.tmp0)

            # self.out_cells1 = []
            # for i in range(Config.LAYERS_STACKED_COUNT):
            #     with tf.variable_scope('RNN_{}'.format(i)):
            #         self.cell1 = _build_rnn_cell()
            #         # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
            #         self.out_cells1.append(self.cell1)
            # self.out_cell1 = tf.nn.rnn_cell.MultiRNNCell(self.out_cells1)
            # self.out_cell1 = tf.contrib.seq2seq.AttentionWrapper(self.out_cell1, self.attention_mechanism)
            # self.out_cell1 = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell1, Config.NUM_OF_CUSTOMERS+1)
            # beam_width = 3
            # tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            #     self.encoder_outputs, multiplier=beam_width)
            # tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
            #     self.encoder_state, multiplier=beam_width)
            # self.tmp4 = tiled_encoder_final_state
            # tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            #     tf.reshape(tf.tile([tf.shape(self.state)[1]], [self.batch_size]), [self.batch_size, 1]), multiplier=beam_width)
            # self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
            #                                                                 memory=tiled_encoder_outputs,
            #                                                                 memory_sequence_length=tiled_sequence_length)
            # self.out_cell1 = tf.contrib.seq2seq.AttentionWrapper(self.out_cell1, self.attention_mechanism)
            # self.out_cell1 = MaskWrapper(self.out_cell1)
            # self.tmp2 = self.out_cell1.zero_state(
            #     dtype=tf.float32, batch_size=self.batch_size * beam_width)
            # self.tmp3 = (self.tmp2[0].clone(
            #     cell_state=tiled_encoder_final_state), self.tmp2[1])

        if Config.DIRECTION == 2:
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                            memory=self.encoder_outputs)
            self.out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
            self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
        if Config.DIRECTION == 3:
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                            memory=self.encoder_outputs)
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
        if Config.DIRECTION == 4:
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                            memory=self.encoder_outputs)
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    # self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            self.out_cell = tf.contrib.seq2seq.AttentionWrapper(self.out_cell, self.attention_mechanism)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            self.beam_out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('Beam_RNN_{}'.format(i)):
                    self.beam_cell = _build_rnn_cell()
                    self.beam_cell = tf.nn.rnn_cell.DropoutWrapper(self.beam_cell, output_keep_prob=self.keep_prob)
                    self.beam_out_cells.append(self.beam_cell)
            self.beam_out_cell = tf.nn.rnn_cell.MultiRNNCell(self.beam_out_cells)
            beam_width = 2
            self.tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                self.encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                self.encoder_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                tf.tile([tf.shape(self.state)[1]], [self.batch_size]), multiplier=beam_width)
            self.beam_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM,
                                                                                 memory=self.tiled_encoder_outputs,
                                                                                 memory_sequence_length=tiled_sequence_length)
            self.beam_out_cell = tf.contrib.seq2seq.AttentionWrapper(self.beam_out_cell, self.beam_attention_mechanism)
            self.beam_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.beam_out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.beam_out_cell = MaskWrapper(self.beam_out_cell)
            self.pred_initial_state = self.beam_out_cell.zero_state(
                dtype=tf.float32, batch_size=self.batch_size * beam_width)
            self.pred_initial_state = self.pred_initial_state.clone(
                cell_state=tiled_encoder_final_state)
            pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                self.beam_out_cell,
                # embedding=lambda sample_ids: tf.gather_nd(
                #     self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                #                            tf.reshape(sample_ids, [-1, 1])], 1)),
                embedding=lambda sample_ids: tf.transpose(
                    tf.reshape(tf.gather_nd(
                        self.state,
                        tf.concat([tf.tile(tf.reshape(tf.range(0, self.batch_size), [-1, 1]), [beam_width, 1]),
                                   tf.reshape(sample_ids, [-1, 1])], 1)), [beam_width, self.batch_size, 2]), [1, 0, 2]),
                # embedding=lambda sample_ids: tf.zeros([5, 2, 2]),
                start_tokens=self.start_tokens,
                end_token=self.end_token,
                initial_state=self.pred_initial_state,
                beam_width=beam_width)
        if Config.DIRECTION == 5:
            cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
                                                     use_peepholes=True,
                                                     output_is_tuple=False,
                                                     state_is_tuple=False)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, Config.NUM_OF_CUSTOMERS+1)
            self.cell = MaskWrapper(self.cell, cell_is_attention=False)
            self.initial_state = self.cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, pred_helper, self.initial_state)

        self.train_final_output, self.train_final_state, train_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            train_decoder, impute_finished=True, maximum_iterations=tf.shape(self.state)[1])
        self.train_final_action = self.train_final_output.sample_id

        self.pred_final_output, self.pred_final_state, pred_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            pred_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
        self.pred_final_action = self.pred_final_output
        # self.pred_final_action = tf.concat([self.pred_final_output.sample_id,
        #                                     tf.zeros([self.batch_size,
        #                                               Config.NUM_OF_CUSTOMERS + 1 - tf.shape(self.pred_final_output.sample_id)[1]],
        #                                              dtype=tf.int32)],
        #                                    1)

        self.weights = tf.sequence_mask(train_final_sequence_lengths, maxlen=Config.NUM_OF_CUSTOMERS+1, dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.train_final_output.rnn_output,
            targets=self.or_action,
            weights=self.weights,
            softmax_loss_function=tf.losses.sparse_softmax_cross_entropy
        )
        tf.summary.scalar("loss", self.loss)

        with tf.name_scope("train"):
            self.lr = tf.train.exponential_decay(
                Config.LEARNING_RATE, self.global_step, 10000,
                .96, staircase=True, name="learning_rate")
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                     global_step=self.global_step,
                                                                     colocate_gradients_with_ops=False)
        tf.summary.scalar("LearningRate", self.lr)
        # for gradient clipping https://github.com/tensorflow/nmt/blob/master/nmt/model.py

        with tf.name_scope("base_line"):
            self.base_line_est = tf.zeros(shape=[self.batch_size, 1])
        # tf.summary.scalar("base_line_est", tf.reduce_mean(self.base_line_est))

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state, current_location):
        feed_dict = {self.state: state, self.current_location: current_location, self.keep_prob: .5,
                     self.or_action: np.zeros((state.shape[0], 9))}
        prediction = self.sess.run([self.train_final_action, self.base_line_est], feed_dict=feed_dict)
        # prediction = (np.zeros([len(state), 9], dtype=np.int32), np.zeros([len(state), 1]))
        return prediction

    def train(self, state, current_location, action, or_action, sampled_cost, or_cost, trainer_id):
        step = self.get_global_step()
        feed_dict = {self.state: state, self.current_location: current_location, self.or_action: or_action,
                     self.sampled_cost: sampled_cost, self.or_cost: or_cost, self.keep_prob: .5}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        if step % 1000 == 0:
            # lambda sample_ids: tf.gather_nd(
                #     self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                #                            tf.reshape(sample_ids, [-1, 1])], 1))
            # print(self.sess.run(self.out_cell, feed_dict=feed_dict))
            # print(self.sess.run(tf.transpose(tf.reshape(tf.gather_nd(self.state, tf.concat([tf.tile(tf.reshape(tf.range(0, self.batch_size), [-1, 1]), [2, 1]), tf.reshape(tf.tile(tf.expand_dims(self.start_tokens+1, 1), [1, 2]), [-1, 1])], 1)), [2, self.batch_size, 2]), [1,0,2]), feed_dict=feed_dict))
            print(self.sess.run([self.or_action, self.train_final_action, self.pred_final_action], feed_dict=feed_dict))
            # print(self.sess.run([self.initial_state], feed_dict=feed_dict))
            if Config.TRAIN:
                summary, _ = self.sess.run([self.merged, self.loss], feed_dict=feed_dict)
                self.log_writer.add_summary(summary, step)
            else:
                self.sess.run([self.loss], feed_dict=feed_dict)
            # print(self.sess.run(self.tmp4, feed_dict=feed_dict)[1])
            print(self.sess.run([self.loss, self.difference_in_length], feed_dict=feed_dict))
            # self.saver.save(self.sess, "./checkpoint/model.ckpt")

    def _create_tensor_board(self):
        # for added metadata https://www.tensorflow.org/programmers_guide/graph_viz
        name = "logs/%s" % self.model_name + time.strftime("_%Y_%m_%d__%H_%M_%s")
        print("Running Model ", name)
        self.log_writer = tf.summary.FileWriter(name)
        self.log_writer.add_graph(self.sess.graph)
        for var in tf.trainable_variables():
            tf.summary.histogram("weights_%s" % var.name, var)
        self.merged = tf.summary.merge_all()

    def finish(self):
        self.log_writer.close()
