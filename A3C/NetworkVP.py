import tensorflow as tf
# import time
from MaskWrapper import MaskWrapper
# from MaskWrapper import MaskWrapperAttnState
# from MaskWrapper import MaskWrapperState
# import numpy as np

from Config import Config


class NetworkVP:
    def __init__(self, device):
        self.device = device

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._create_graph()
            if Config.GPU == 1:
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = .95
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            self.sess = tf.Session(
                graph=self.graph,
                config=config
            )
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            self.name = Config.MODEL_NAME
            # self.name = 'TSP_' + str(Config.NUM_OF_CUSTOMERS) + '_dir_' + str(Config.DIRECTION) + \
            #             '_EncEmb' + str(Config.ENC_EMB) + '_DecEmb' + str(Config.DEC_EMB) + \
            #             '_Drop' + str(Config.DROPOUT) + '_MaxGrad_' + str(Config.MAX_GRAD) + \
            #             '_BnF_GPU_' + str(Config.GPU) + '_LogitPen_' + str(Config.LOGIT_PENALTY) + \
            #             '_NewTrainHelper' + \
            #             time.strftime("_%Y_%m_%d__%H_%M_%s")
            print("Running Model ", self.name)
            if Config.TRAIN == 1:
                self._create_tensor_board()
            if Config.RESTORE == 1:
                print("Restoring Parameters from latest checkpoint:")
                latest_checkpoint = tf.train.latest_checkpoint(str(Config.PATH) + 'checkpoint/' + Config.MODEL_NAME + '/')
                print(latest_checkpoint)
                self.saver.restore(self.sess, latest_checkpoint)

    def _model_save(self):
        self.saver.save(self.sess, str(Config.PATH) + 'checkpoint/' + self.name + '/model.ckpt')

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
        self.relative_length = tf.reduce_mean(self.sampled_cost/self.or_cost)
        self.start_tokens = tf.placeholder(tf.int32, shape=[None])
        self.end_token = -1
        self.is_training = tf.placeholder_with_default(
            tf.constant(True, dtype=tf.bool),
            shape=(), name='is_training'
        )
        with tf.name_scope("Config"):
            tf.summary.scalar("DIRECTION", Config.DIRECTION)
            tf.summary.scalar("batch_size", self.batch_size)
            tf.summary.scalar("Config.LAYERS_STACKED_COUNT", Config.LAYERS_STACKED_COUNT)
            tf.summary.scalar("RNN_HIDDEN_DIM", Config.RNN_HIDDEN_DIM)
            tf.summary.scalar("RUN_TIME", Config.RUN_TIME)
            tf.summary.scalar("LOGIT_CLIP_SCALAR", Config.LOGIT_CLIP_SCALAR)
            tf.summary.scalar("MAX_GRAD", Config.MAX_GRAD)
            tf.summary.scalar("NUM_OF_CUSTOMERS", Config.NUM_OF_CUSTOMERS)
            tf.summary.scalar("EncEmb", tf.cast(Config.ENC_EMB, tf.int32))
            tf.summary.scalar("DecEmb", tf.cast(Config.DEC_EMB, tf.int32))
            tf.summary.scalar("Droput", tf.cast(Config.DROPOUT, tf.int32))
            tf.summary.scalar("MaxGrad", Config.MAX_GRAD)
            tf.summary.scalar("LogitPen", Config.LOGIT_PENALTY)
            tf.summary.scalar("LogitClipScalar", Config.LOGIT_CLIP_SCALAR)
            tf.summary.scalar("GPU", Config.GPU)

        if Config.ENC_EMB == 1:
            W_embed = tf.get_variable("weights", [1, 2, Config.RNN_HIDDEN_DIM], initializer=tf.contrib.layers.xavier_initializer())
            self.enc_inputs = tf.nn.conv1d(self.state, W_embed, 1, "VALID", name="embedded_input")
            # self.enc_inputs = tf.layers.batch_normalization(self.enc_inputs, axis=2, training=self.is_training, reuse=None)
        else:
            self.enc_inputs = self.state

        def _build_rnn_cell():
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
            cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
            if Config.DROPOUT == 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                     input_keep_prob=self.keep_prob,
                                                     output_keep_prob=self.keep_prob,
                                                     state_keep_prob=self.keep_prob)
            return cell

        def _build_attention(cell, memory):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM, memory=memory)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, output_attention=False)
            return attn_cell

        ########## ENCODER ##########
        if Config.DIRECTION == 1:
            self.in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
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
                    self.in_cells.append(self.cell)
            (self.encoder_outputs, self.encoder_fw_state, self.encoder_bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                             cells_fw=self.in_cells, cells_bw=self.in_cells, inputs=self.enc_inputs, dtype=tf.float32)
            self.encoder_state = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                encoder_state_c = tf.concat(
                    (self.encoder_fw_state[i].c, self.encoder_bw_state[i].c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (self.encoder_fw_state[i].h, self.encoder_bw_state[i].h), 1, name='bidirectional_concat_h')
                self.encoder_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
            self.encoder_state = tuple(self.encoder_state)
        if Config.DIRECTION == 4:
            self.in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    self.in_cells.append(self.cell)
            self.in_cell = tf.nn.rnn_cell.MultiRNNCell(self.in_cells)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.in_cell,
                                                                         self.enc_inputs,
                                                                         dtype=tf.float32)

        ########## HELPERS ##########
        self.gather_ids = tf.concat([tf.expand_dims(
            tf.reshape(tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), [1, tf.shape(self.state)[1]]), [-1]), -1),
                                     tf.reshape(self.or_action, [-1, 1])], -1)
        self.training_inputs = tf.reshape(tf.gather_nd(self.state, self.gather_ids), [self.batch_size, tf.shape(self.state)[1], 2])
        self.training_inputs = tf.concat([tf.zeros([self.batch_size, 1, 2]), self.training_inputs], axis=1)
        self.training_inputs = self.training_inputs[:, :-1, :]

        def embed_fn(sample_ids):
            return(tf.reshape(tf.nn.conv1d(
                    tf.reshape(tf.gather_nd(self.state, tf.concat([tf.reshape(tf.range(self.batch_size), [-1, 1]),
                                                                   tf.reshape(sample_ids, [-1, 1])], -1)), [self.batch_size, 1, 2]),
                    W_embed, 1, "VALID", name="embedded_input"), [self.batch_size, Config.RNN_HIDDEN_DIM]))
        if Config.DEC_EMB == 1:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda sample_ids: embed_fn(sample_ids),
                self.start_tokens,
                self.end_token)
            self.training_inputs = tf.nn.conv1d(self.training_inputs, W_embed, 1, "VALID", name="embedded_input")
        else:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda sample_ids: tf.gather_nd(
                    self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
                                           tf.reshape(sample_ids, [-1, 1])], 1)
                ),
                self.start_tokens,
                self.end_token)
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.training_inputs,
                                                         tf.fill([self.batch_size], Config.NUM_OF_CUSTOMERS+1))

        # self.training_inputs = tf.expand_dims(tf.gather_nd(self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
        #                                                                          tf.reshape(self.or_action[:, 0], [-1, 1])], 1)), 1)
        # for i in range(1, Config.NUM_OF_CUSTOMERS+1):
        #     self.training_inputs = tf.concat(
        #         [self.training_inputs, tf.expand_dims(
        #             tf.gather_nd(self.state, tf.concat([tf.reshape(tf.range(0, self.batch_size), [-1, 1]),
        #                                                 tf.reshape(self.or_action[:, i], [-1, 1])], 1)), 1)], 1)

        ########## DECODER ##########
        if Config.DIRECTION == 1:
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            self.out_cell = _build_attention(self.out_cell, self.encoder_outputs)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)

            self.critic_out_cells = [_build_rnn_cell()]
            self.critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(self.critic_out_cells)
            self.critic_out_cell = _build_attention(self.critic_out_cell, self.encoder_outputs)
            self.critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.critic_out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.critic_out_cell = MaskWrapper(self.critic_out_cell)
            self.critic_initial_state = self.critic_out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.critic_initial_state = self.critic_initial_state.clone(cell_state=self.encoder_state)

            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
            critic_decoder = tf.contrib.seq2seq.BasicDecoder(self.critic_out_cell, pred_helper, self.critic_initial_state)
        if Config.DIRECTION == 2:
            self.out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
            self.out_cell = _build_attention(self.out_cell, self.encoder_outputs)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)

            self.critic_out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
            # self.critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(self.critic_out_cells)
            self.critic_out_cell = _build_attention(self.critic_out_cell, self.encoder_outputs)
            self.critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.critic_out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.critic_out_cell = MaskWrapper(self.critic_out_cell)
            self.critic_initial_state = self.critic_out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.critic_initial_state = self.critic_initial_state.clone(cell_state=self.encoder_state)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
            critic_decoder = tf.contrib.seq2seq.BasicDecoder(self.critic_out_cell, pred_helper, self.critic_initial_state)
        if Config.DIRECTION == 3:
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            self.out_cell = _build_attention(self.out_cell, self.encoder_outputs)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.out_cell = MaskWrapper(self.out_cell)
            self.initial_state = self.out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)

            self.critic_out_cells = [tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)]
            self.critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(self.critic_out_cells)
            self.critic_out_cell = _build_attention(self.critic_out_cell, self.encoder_outputs)
            self.critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.critic_out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.critic_out_cell = MaskWrapper(self.critic_out_cell)
            self.critic_initial_state = self.critic_out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.critic_initial_state = self.critic_initial_state.clone(cell_state=self.encoder_state)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.out_cell, pred_helper, self.initial_state)
            critic_decoder = tf.contrib.seq2seq.BasicDecoder(self.critic_out_cell, pred_helper, self.critic_initial_state)
        if Config.DIRECTION == 4:
            self.critic_out_cells = [_build_rnn_cell()]
            self.critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(self.critic_out_cells)
            self.critic_out_cell = _build_attention(self.critic_out_cell, self.encoder_outputs)
            self.critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.critic_out_cell, Config.NUM_OF_CUSTOMERS+1)
            self.critic_out_cell = MaskWrapper(self.critic_out_cell)
            self.critic_initial_state = self.critic_out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            self.critic_initial_state = self.critic_initial_state.clone(cell_state=self.encoder_state)
            critic_decoder = tf.contrib.seq2seq.BasicDecoder(self.critic_out_cell, pred_helper, self.critic_initial_state)
            self.out_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('RNN_{}'.format(i)):
                    self.cell = _build_rnn_cell()
                    self.out_cells.append(self.cell)
            self.out_cell = tf.nn.rnn_cell.MultiRNNCell(self.out_cells)
            with tf.variable_scope("beam"):
                self.train_out_cell = _build_attention(self.out_cell, self.encoder_outputs)
                self.train_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.train_out_cell, Config.NUM_OF_CUSTOMERS+1)
                self.train_out_cell = MaskWrapper(self.train_out_cell)
                self.initial_state = self.train_out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                self.initial_state = self.initial_state.clone(cell_state=self.encoder_state)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(self.train_out_cell, train_helper, self.initial_state)
            with tf.variable_scope("beam", reuse=True):
                beam_width = 10
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                    self.encoder_outputs, multiplier=beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_state, multiplier=beam_width)
                # tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                #     tf.tile([tf.shape(self.state)[1]], [self.batch_size]), multiplier=beam_width)
                self.pred_out_cell = _build_attention(self.out_cell, tiled_encoder_outputs)
                self.pred_out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.pred_out_cell, Config.NUM_OF_CUSTOMERS+1)
                self.pred_out_cell = MaskWrapper(self.pred_out_cell)
                self.pred_initial_state = self.pred_out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size * beam_width)
                self.pred_initial_state = self.pred_initial_state.clone(
                    cell_state=tiled_encoder_final_state)
                if Config.DEC_EMB == 1:
                    def beam_embed(x):
                        return(
                            tf.reshape(tf.nn.conv1d(tf.reshape(tf.gather_nd(self.state, tf.concat(
                                [tf.reshape(tf.range(self.batch_size), [-1, 1]),
                                 tf.reshape(x, [-1, 1])], -1)), [self.batch_size, 1, 2]), W_embed, 1, "VALID"),
                                       [self.batch_size, Config.RNN_HIDDEN_DIM])
                        )
                    # even with logit pen it is outputing infeas routes
                    pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        self.pred_out_cell,
                        embedding=lambda ids: tf.transpose(
                            tf.map_fn(beam_embed, tf.transpose(ids, [1, 0]), dtype=tf.float32),
                            [1, 0, 2]),
                        start_tokens=self.start_tokens,
                        end_token=self.end_token,
                        initial_state=self.pred_initial_state,
                        beam_width=beam_width)
                else:
                    pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        self.pred_out_cell,
                        embedding=lambda sample_ids: tf.transpose(
                            tf.reshape(tf.gather_nd(
                                self.state,
                                tf.concat([tf.tile(tf.reshape(tf.range(0, self.batch_size), [-1, 1]), [beam_width, 1]),
                                           tf.reshape(sample_ids, [-1, 1])], 1)), [beam_width, self.batch_size, 2]), [1, 0, 2]),
                        start_tokens=self.start_tokens,
                        end_token=self.end_token,
                        initial_state=self.pred_initial_state,
                        beam_width=beam_width)
                # pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.train_out_cell, pred_helper, self.initial_state)
        if Config.DIRECTION == 5:
            cell = tf.contrib.grid_rnn.Grid2LSTMCell(Config.RNN_HIDDEN_DIM,
                                                     use_peepholes=True,
                                                     output_is_tuple=False,
                                                     state_is_tuple=False)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, Config.NUM_OF_CUSTOMERS+1)
            self.cell = MaskWrapper(self.cell, cell_is_attention=False)
            self.initial_state = self.cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, train_helper, self.initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, pred_helper, self.initial_state)
        if Config.DIRECTION == 6:
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
        if Config.DIRECTION == 7:
            self.cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
                                                          use_peepholes=True,
                                                          output_is_tuple=False,
                                                          state_is_tuple=False)
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, Config.NUM_OF_CUSTOMERS+1)
            self.cell = MaskWrapper(self.cell, cell_is_attention=False)
            self.initial_state = self.cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, train_helper, self.initial_state,
                                                            output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, pred_helper, self.initial_state,
                                                           output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))

        self.train_final_output, self.train_final_state, train_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            train_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
        self.train_final_action = self.train_final_output.sample_id

        self.pred_final_output, self.pred_final_state, pred_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            pred_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
        if Config.DIRECTION == 4:
            self.pred_final_action = tf.transpose(self.pred_final_output.predicted_ids, [2, 0, 1])[0]
        else:
            self.pred_final_action = self.pred_final_output.sample_id

        self.critic_final_output, self.critic_final_state, critic_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            critic_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
        if Config.DIRECTION != 2:
            self.base_line_est = tf.layers.dense(self.critic_final_state.cell_state[0].h, Config.RNN_HIDDEN_DIM)
        else:
            self.base_line_est = tf.layers.dense(self.critic_final_state.cell_state.c, Config.RNN_HIDDEN_DIM, activation=tf.nn.relu)
        self.base_line_est = tf.layers.dense(self.base_line_est, 1)
        self.critic_loss = tf.losses.mean_squared_error(self.or_cost, self.base_line_est)

        tf.summary.histogram("LocationStartDist", tf.transpose(self.pred_final_action, [1, 0])[0])
        tf.summary.histogram("LocationEndDist", tf.transpose(self.pred_final_action, [1, 0])[-1])

        # self.pred_final_action = tf.concat([self.pred_final_action,
        #                                     tf.zeros([self.batch_size,
        #                                               Config.NUM_OF_CUSTOMERS + 1 - tf.shape(self.pred_final_action)[1]],
        #                                              dtype=tf.int32)],
        #                                    1)

        # self.weights = tf.sequence_mask(train_final_sequence_lengths, maxlen=Config.NUM_OF_CUSTOMERS+1, dtype=tf.float32)

        self.weights = tf.ones([self.batch_size, tf.shape(self.state)[1]])
        # if Config.DIRECTION == 4:
        #     self.logits = self.train_final_output.rnn_output
        # else:
        #     self.logits = self.pred_final_output.rnn_output

        self.logits = self.train_final_output.rnn_output
        if Config.LOGIT_CLIP_SCALAR != 0:
            self.logits = Config.LOGIT_CLIP_SCALAR*tf.nn.tanh(self.logits)

        if Config.REINFORCE == 0:
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.or_action,
                weights=self.weights
            )
        else:
            self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.train_final_action)
            self.loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.neg_log_prob, axis=1), self.sampled_cost-self.base_line_est))

        with tf.name_scope("train"):
            if Config.GPU == 1:
                colocate = True
            else:
                colocate = False
            self.lr = tf.train.exponential_decay(
                Config.LEARNING_RATE, self.global_step, 10000,
                .9, staircase=True, name="learning_rate")
            self.critic_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss, global_step=self.global_step)
            # self.train_op = tf.train.AdadeltaOptimizer(Config.LEARNING_RATE).minimize(self.loss,
            #                                                                           global_step=self.global_step,
            #                                                                           colocate_gradients_with_ops=True)
            if Config.MAX_GRAD != 0:
                params = tf.trainable_variables()
                self.gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=colocate)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(self.gradients, Config.MAX_GRAD)
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
                tf.summary.scalar("grad_norm", gradient_norm)
                tf.summary.scalar("LearningRate", self.lr)
            else:
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                         global_step=self.global_step,
                                                                         colocate_gradients_with_ops=colocate)
        # for gradient clipping https://github.com/tensorflow/nmt/blob/master/nmt/model.py

        with tf.name_scope("loss"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("critic_loss", self.critic_loss)
        with tf.name_scope("Performace"):
            tf.summary.scalar("Relative Critic Loss", tf.reduce_mean(self.base_line_est/self.or_cost))
            tf.summary.scalar("difference_in_length", self.difference_in_length)
            tf.summary.scalar("relative_length", self.relative_length)
        # self.base_line_est = tf.zeros(shape=[self.batch_size, 1])

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state, current_location, depot_idx):
        feed_dict = {self.state: state, self.current_location: current_location, self.is_training: False,
                     self.keep_prob: 1.0, self.start_tokens: depot_idx}
        prediction = self.sess.run([self.pred_final_action, self.base_line_est], feed_dict=feed_dict)
        # prediction = [np.zeros([state.shape[0], Config.NUM_OF_CUSTOMERS+1], dtype=np.int32),
        #               np.zeros([state.shape[0], 1], dtype=np.float32)]
        return prediction

    def train(self, state, current_location, action, or_action, sampled_cost, or_cost, depot_idx, trainer_id):
        step = self.get_global_step()
        feed_dict = {self.state: state, self.current_location: current_location, self.or_action: or_action,
                     self.sampled_cost: sampled_cost, self.or_cost: or_cost, self.start_tokens: depot_idx, self.keep_prob: .8}
        # print("step", step)
        print("or_action")
        print(self.sess.run([self.or_action], feed_dict=feed_dict))
        print("train_action")
        print(self.sess.run([self.pred_final_action], feed_dict=feed_dict))
        print("or_cost")
        print(self.sess.run([self.or_cost], feed_dict=feed_dict))
        print("sampled_cost")
        print(self.sess.run([self.sampled_cost], feed_dict=feed_dict))
        # print(self.sess.run([tf.nn.softmax(self.logits)], feed_dict=feed_dict))
        # print(self.sess.run([self.train_final_action], feed_dict=feed_dict))
        # print(self.sess.run([], feed_dict=feed_dict))
        # print("train_output")
        # print(self.sess.run([self.train_final_output.sample_id], feed_dict=feed_dict))
        # print("pred_output")
        # print(self.sess.run([self.pred_final_action], feed_dict=feed_dict))
        # print("loss")
        # print(self.sess.run([self.loss], feed_dict=feed_dict))
        # print()
        # print(self.sess.run([self.state], feed_dict=feed_dict))
        # print(self.sess.run([self.tmp], feed_dict=feed_dict))
        if step % 100 == 0:
            if Config.TRAIN == 1:
                _, _, summary, loss, diff = self.sess.run([self.train_op, self.critic_train_op,
                                                           self.merged, self.loss, self.relative_length], feed_dict=feed_dict)
                self.log_writer.add_summary(summary, step)
            else:
                _, _, loss, diff = self.sess.run([self.train_op, self.critic_train_op,
                                                  self.loss, self.relative_length], feed_dict=feed_dict)
            print(loss, diff)
        else:
            self.sess.run(self.train_op, feed_dict=feed_dict)
        if step % 1000 == 0 and Config.TRAIN:
            print("Saving Model...")
            self._model_save()
            # print(self.sess.run([self.pred_final_action], feed_dict=feed_dict))
            # print(self.sess.run([self.or_action], feed_dict=feed_dict))

    def _create_tensor_board(self):
        # for added metadata https://www.tensorflow.org/programmers_guide/graph_viz
        log_name = str(Config.PATH) + "logs/" + self.name
        self.log_writer = tf.summary.FileWriter(log_name)
        self.log_writer.add_graph(self.sess.graph)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram("weights_%s" % var.name, var)
        self.merged = tf.summary.merge_all()

    def finish(self):
        if Config.TRAIN == 1:
            self._model_save()
        self.log_writer.close()
