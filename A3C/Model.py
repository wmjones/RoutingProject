import tensorflow as tf
from Config import Config
from MyWrapper import MaskWrapper, MaskWrapperAttnState
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops

def _build_rnn_cell(keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
    # , initializer=tf.contrib.layers.xavier_initializer())
    # cell = tf.nn.rnn_cell.GRUCell(Config.RNN_HIDDEN_DIM)
    if Config.DROPOUT == 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell


def _build_attention(cell, memory, probability_fn=None):
    if Config.USE_BAHDANAU == 0:
        if Config.DIRECTION == 2 or Config.DIRECTION == 3:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=Config.RNN_HIDDEN_DIM*2, memory=memory,
                                                                    probability_fn=probability_fn, scale=True)
        else:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=Config.RNN_HIDDEN_DIM, memory=memory,
                                                                    probability_fn=probability_fn, scale=True)
    else:
        if Config.DIRECTION == 2 or Config.DIRECTION == 3:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM*2, memory=memory,
                                                                       probability_fn=probability_fn, normalize=True)
        else:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM, memory=memory,
                                                                       probability_fn=probability_fn, normalize=True)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, output_attention=True)
    return attn_cell


def Encoder(enc_inputs, keep_prob):
    with tf.variable_scope("Encoder"):
        if Config.DIRECTION == 1:
            in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('ENC_RNN_{}'.format(i)):
                    enc_cell = _build_rnn_cell(keep_prob)
                    in_cells.append(enc_cell)
            in_cell = tf.nn.rnn_cell.MultiRNNCell(in_cells)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(in_cell,
                                                               enc_inputs,
                                                               dtype=tf.float32)
        if Config.DIRECTION == 2:
            in_cell = _build_rnn_cell(keep_prob)
            (bi_outputs, (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                             cell_fw=in_cell, cell_bw=in_cell, inputs=enc_inputs, dtype=tf.float32)
            encoder_state_c = tf.concat(
                (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
            encoder_state_h = tf.concat(
                (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            encoder_outputs = tf.concat(bi_outputs, -1)
        if Config.DIRECTION == 3:
            in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('ENC_RNN_{}'.format(i)):
                    cell = _build_rnn_cell(keep_prob)
                    in_cells.append(cell)
            (encoder_outputs, encoder_fw_state, encoder_bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                             cells_fw=in_cells, cells_bw=in_cells, inputs=enc_inputs, dtype=tf.float32)
            encoder_state = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                encoder_state_c = tf.concat(
                    (encoder_fw_state[i].c, encoder_bw_state[i].c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state[i].h, encoder_bw_state[i].h), 1, name='bidirectional_concat_h')
                encoder_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
            encoder_state = tuple(encoder_state)
        return encoder_outputs, encoder_state


def Helper(problem_state, batch_size, training_inputs, start_tokens, end_token):
    with tf.variable_scope("Helper"):
        if Config.STOCHASTIC == 1:
            def initialize_fn():
                if Config.STATE_EMBED == 0:
                    return (tf.tile([False], [batch_size]), tf.zeros([batch_size, 2]))
                else:
                    return (tf.tile([False], [batch_size]), tf.zeros([batch_size, Config.RNN_HIDDEN_DIM]))

            def sample_fn(time, outputs, state):
                logits = outputs * Config.INVERSE_SOFTMAX_TEMP
                sample_id_sampler = categorical.Categorical(logits=logits)
                sample_ids = sample_id_sampler.sample()
                return sample_ids

            def next_inputs_fn(time, outputs, state, sample_ids):
                finished = tf.tile([tf.equal(time, Config.NUM_OF_CUSTOMERS)], [batch_size])
                next_inputs = tf.gather_nd(
                    problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                              tf.reshape(sample_ids, [-1, 1])], 1)
                )
                next_state = MaskWrapperAttnState(
                    AttnState=state.AttnState,
                    mask=state.mask + tf.one_hot(sample_ids, depth=Config.NUM_OF_CUSTOMERS, dtype=tf.float32))
                return (finished, next_inputs, next_state)

            train_helper = tf.contrib.seq2seq.CustomHelper(
                initialize_fn=initialize_fn,
                sample_fn=sample_fn,
                next_inputs_fn=next_inputs_fn
            )
            pred_helper = train_helper
        if Config.REINFORCE == 1 and Config.STOCHASTIC == 0:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    lambda sample_ids: tf.gather_nd(
                        problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                                  tf.reshape(sample_ids, [-1, 1])], 1)
                    ),
                    start_tokens,
                    end_token)
            train_helper = pred_helper
        if Config.REINFORCE == 0 and Config.STOCHASTIC == 0:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda sample_ids: tf.gather_nd(
                    problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                              tf.reshape(sample_ids, [-1, 1])], 1)
                ),
                start_tokens,
                end_token)
            train_helper = tf.contrib.seq2seq.TrainingHelper(training_inputs,
                                                             tf.fill([batch_size], Config.NUM_OF_CUSTOMERS))
            # train_helper = pred_helper
        return train_helper, pred_helper


def Decoder(batch_size, encoder_state, encoder_outputs, train_helper, pred_helper, problem_state,
            start_tokens, end_token, keep_prob, raw_state, DECODER_TYPE):
    if Config.DIRECTION == 1:
        out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('DEC_RNN_{}'.format(i)):
                cell = _build_rnn_cell(keep_prob)
                out_cells.append(cell)
        out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
        out_cell = _build_attention(out_cell, encoder_outputs)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, Config.NUM_OF_CUSTOMERS)
        out_cell = MaskWrapper(out_cell)

        # critic_out_cells = []
        # for i in range(Config.LAYERS_STACKED_COUNT):
        #     with tf.variable_scope('CRITIC_RNN_{}'.format(i)):
        #         cell = _build_rnn_cell(keep_prob)
        #         critic_out_cells.append(cell)
        # critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        # critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        # critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        # critic_out_cell = MaskWrapper(critic_out_cell)
        # critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        # critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))
        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        # critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    if Config.DIRECTION == 2:
        out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
        out_cell = _build_attention(out_cell, encoder_outputs)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, Config.NUM_OF_CUSTOMERS)
        out_cell = MaskWrapper(out_cell)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))

        # critic_out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
        # critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        # critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        # critic_out_cell = MaskWrapper(critic_out_cell)
        # critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        # critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        # critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    if Config.DIRECTION == 3:
        out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('DEC_RNN_{}'.format(i)):
                cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
                out_cells.append(cell)
        out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
        out_cell = _build_attention(out_cell, encoder_outputs)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, Config.NUM_OF_CUSTOMERS)
        out_cell = MaskWrapper(out_cell)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))

        # critic_out_cells = []
        # for i in range(Config.LAYERS_STACKED_COUNT):
        #     with tf.variable_scope('CRITIC_DEC_RNN_{}'.format(i)):
        #         cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
        #         critic_out_cells.append(cell)
        # critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        # critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        # critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        # critic_out_cell = MaskWrapper(critic_out_cell)
        # critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        # critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        # critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    # if Config.DIRECTION == 5:
    #     cell = tf.contrib.grid_rnn.Grid2LSTMCell(Config.RNN_HIDDEN_DIM,
    #                                              use_peepholes=True,
    #                                              output_is_tuple=False,
    #                                              state_is_tuple=False)
    #     cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
    #     cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
    #     cell = MaskWrapper(cell, cell_is_attention=False)
    #     initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    #     train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state)
    #     pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state)
    # if Config.DIRECTION == 6:
    #     cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
    #                                              use_peepholes=True,
    #                                              output_is_tuple=False,
    #                                              state_is_tuple=False)
    #     cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
    #     cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
    #     cell = MaskWrapper(cell, cell_is_attention=False)
    #     initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    #     train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state)
    #     pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state)
    # if Config.DIRECTION == 7:
    #     cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
    #                                              use_peepholes=True,
    #                                              output_is_tuple=False,
    #                                              state_is_tuple=False)
    #     cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
    #     cell = MaskWrapper(cell, cell_is_attention=False)
    #     initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    #     train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state,
    #                                                     output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))
    #     pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state,
    #                                                    output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))
    if Config.DIRECTION == 4:
        out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                cell = _build_rnn_cell(keep_prob)
                out_cells.append(cell)
        out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
        out_cell = _build_attention(out_cell, encoder_outputs)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, Config.NUM_OF_CUSTOMERS)
        out_cell = MaskWrapper(out_cell)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        # critic_out_cells = [_build_rnn_cell(keep_prob)]
        # critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        # critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        # critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        # critic_out_cell = MaskWrapper(critic_out_cell)
        # critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        # critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    if Config.DIRECTION == 5:
        out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                cell = _build_rnn_cell(keep_prob)
                out_cells.append(cell)
        out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
        out_cell = _build_attention(out_cell, encoder_outputs, probability_fn=tf.identity)
        out_cell = MaskWrapper(out_cell)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)

    with tf.variable_scope("Conv_Critic"):
        out = raw_state
        for i in range(5):
            out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, 2, padding="SAME", activation=tf.nn.relu)
        out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, Config.NUM_OF_CUSTOMERS+1)
        out = tf.reshape(out, [-1, Config.RNN_HIDDEN_DIM])
        critic_network_pred = tf.layers.dense(tf.layers.dense(out, 10, tf.nn.relu), 1)
    return train_decoder, pred_decoder, critic_network_pred


def Reza_Model(batch_size, problem_state):
    def Dist_mat(A):
        nnodes = tf.shape(A)[1]
        A1 = tf.tile(tf.expand_dims(A, 1), [1, nnodes, 1, 1])
        A2 = tf.tile(tf.expand_dims(A, 2), [1, 1, nnodes, 1])
        dist = tf.norm(A1-A2, axis=3)
        return dist

    def Actor(emb_inp, decoder_input, decoder_state, mask, cell):
        emb_inp = emb_inp[:, :-1, :]
        with tf.variable_scope("Decoder"):
            with tf.variable_scope("LSTM"):
                _, decoder_state = tf.nn.dynamic_rnn(
                    cell,
                    decoder_input,
                    initial_state=decoder_state)
            hy = decoder_state[0][1]
            for j in range(1):
                with tf.variable_scope("Glimpse"+str(j)):
                    v = tf.get_variable('v', [1, Config.RNN_HIDDEN_DIM],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    vBLOCK = tf.expand_dims(tf.tile(v, [batch_size, 1]), 1)
                    e = tf.tile(tf.transpose(tf.layers.conv1d(emb_inp, Config.RNN_HIDDEN_DIM, 1),
                                             perm=[0, 2, 1]), [1, 1, 1])
                    q = tf.tile(tf.expand_dims(tf.layers.dense(hy, Config.RNN_HIDDEN_DIM), 2), [1, 1, tf.shape(e)[2]])
                    u = tf.squeeze(tf.matmul(vBLOCK, tf.nn.tanh(e + q)), 1)
                    prob = tf.nn.softmax(u)
                    hy = tf.squeeze(tf.matmul(e, tf.expand_dims(prob, 2)), 2)
        with tf.variable_scope("Attention"):
            v = tf.get_variable('v', [1, Config.RNN_HIDDEN_DIM],
                                initializer=tf.contrib.layers.xavier_initializer())
            vBLOCK = tf.expand_dims(tf.tile(v, [batch_size, 1]), 1)
            e = tf.tile(tf.transpose(tf.layers.conv1d(emb_inp, Config.RNN_HIDDEN_DIM, 1),
                                     perm=[0, 2, 1]), [1, 1, 1])
            q = tf.tile(tf.expand_dims(tf.layers.dense(hy, Config.RNN_HIDDEN_DIM), 2), [1, 1, tf.shape(e)[2]])
            u = tf.squeeze(tf.matmul(vBLOCK, tf.nn.tanh(e + q)), 1)
            logit = u - tf.cast(1e6*mask, tf.float32)
            prob = tf.nn.softmax(logit)
        return prob, decoder_state, logit

    with tf.variable_scope("Embedding"):
        if Config.REZA == 0:
            emb_inp = tf.layers.conv1d(problem_state, Config.RNN_HIDDEN_DIM, 1)
        else:
            dist = Dist_mat(problem_state)
            emb_inp = tf.layers.conv1d(dist, Config.RNN_HIDDEN_DIM, 1)
    with tf.variable_scope("Actor"):
        initial_state = tf.zeros([1, 2, batch_size, Config.RNN_HIDDEN_DIM])
        tmp = tf.unstack(initial_state, axis=0)
        rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(tmp[idx][0], tmp[idx][1]) for idx in range(1)])
        decoder_state = rnn_tuple_state
        decoder_input = tf.get_variable('decoder_input', [1, 1, Config.RNN_HIDDEN_DIM],
                                        initializer=tf.contrib.layers.xavier_initializer())
        decoder_input = tf.tile(decoder_input, [batch_size, 1, 1])
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(Config.RNN_HIDDEN_DIM)])
        mask = tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS], dtype=tf.float32)
        probs = []
        masks = []
        idxs = []
        actions_tmp = []
        logprobs = []
        logits = []
        BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)
        for i in range(Config.NUM_OF_CUSTOMERS):
            prob, decoder_state, logit = Actor(emb_inp, decoder_input, decoder_state, mask, cell)
            probs.append(prob)
            tf.get_variable_scope().reuse_variables()
            idx = tf.expand_dims(tf.argmax(prob, 1), 1)
            mask = mask + tf.one_hot(tf.squeeze(idx, 1), Config.NUM_OF_CUSTOMERS)
            batched_idx = tf.concat([BatchSequence, idx], 1)
            masks.append(mask)
            decoder_input = tf.expand_dims(tf.gather_nd(tf.tile(emb_inp, [1, 1, 1]), batched_idx), 1)
            action = tf.gather_nd(tf.tile(problem_state, [1, 1, 1]), batched_idx)
            logprob = tf.log(tf.gather_nd(prob, batched_idx))
            idxs.append(idx)
            actions_tmp.append(action)
            logprobs.append(logprob)
            logits.append(logit)
        pred_final_action = tf.convert_to_tensor(idxs)
        pred_final_action = tf.transpose(tf.reshape(
            pred_final_action, [Config.NUM_OF_CUSTOMERS, batch_size]),
                                              [1, 0])
        train_final_action = pred_final_action
        logits = tf.convert_to_tensor(logits)
        logits = tf.transpose(tf.reshape(
            logits, [Config.NUM_OF_CUSTOMERS, batch_size, -1]),
                                              [1, 0, 2])

    with tf.variable_scope("Critic"):
        with tf.variable_scope("Encoder"):
            initial_state = tf.zeros([1, 2, batch_size, Config.RNN_HIDDEN_DIM])
            tmp = tf.unstack(initial_state, axis=0)
            rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(tmp[idx][0], tmp[idx][1])
                                     for idx in range(1)])
            hy = rnn_tuple_state[0][1]  # seems like a really bad way to create a tensor of zeros

        with tf.variable_scope("Process"):
            for i in range(3):
                with tf.variable_scope("P"+str(i)):
                    v = tf.get_variable('v', [1, Config.RNN_HIDDEN_DIM], initializer=tf.contrib.layers.xavier_initializer())
                    vBLOCK = tf.expand_dims(tf.tile(v, [batch_size, 1]), 1)
                    e = tf.transpose(tf.layers.conv1d(emb_inp, Config.RNN_HIDDEN_DIM, 1), perm=[0, 2, 1])
                    q = tf.tile(tf.expand_dims(tf.layers.dense(hy, Config.RNN_HIDDEN_DIM), 2), [1, 1, tf.shape(e)[2]])
                    u = tf.squeeze(tf.matmul(vBLOCK, tf.nn.tanh(e + q)), 1)
                    prob = tf.nn.softmax(u)
                    hy = tf.squeeze(tf.matmul(e, tf.expand_dims(prob, 2)), 2)

        with tf.variable_scope("Linear"):
            base_line_est = tf.squeeze(tf.layers.dense(tf.layers.dense(hy, Config.RNN_HIDDEN_DIM, tf.nn.relu), 1), 1)
            base_line_est = tf.reshape(base_line_est, [-1, 1])
    return train_final_action, pred_final_action, base_line_est, logits


def Wyatt_Model(batch_size, problem_state, raw_state):
    if Config.STATE_EMBED == 1:
        if Config.INPUT_TIME == 0:
            initial_inputs = tf.zeros([batch_size, Config.RNN_HIDDEN_DIM])
        else:
            initial_inputs = tf.zeros([batch_size, Config.RNN_HIDDEN_DIM+1])
    else:
        if Config.INPUT_TIME == 0:
            initial_inputs = tf.zeros([batch_size, 2])
        else:
            initial_inputs = tf.zeros([batch_size, 3])
    if Config.INPUT_ALL == 1:
        initial_inputs = tf.reshape(raw_state, [batch_size, 2*(Config.NUM_OF_CUSTOMERS+1)])

    with tf.variable_scope("Actor"):
        cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
        cell = _build_attention(cell, problem_state, tf.identity)
        state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        mask = tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS], dtype=tf.float32)
        actions = []
        logits = []
        inputs = initial_inputs
        for i in range(Config.NUM_OF_CUSTOMERS):
            outputs, state = cell(inputs, state)
            logit = state.alignments-mask*Config.LOGIT_PENALTY
            action = tf.argmax(logit, axis=1, output_type=tf.int32)
            mask = mask + tf.one_hot(action, Config.NUM_OF_CUSTOMERS, dtype=tf.float32)
            if Config.INPUT_TIME == 0:
                inputs = tf.gather_nd(problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                                                tf.reshape(action, [-1, 1])], 1))
            else:
                inputs = tf.concat([tf.to_float(tf.reshape(tf.tile([i], [batch_size]), [-1, 1])),
                                    tf.gather_nd(problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                                                           tf.reshape(action, [-1, 1])], 1))], 1)
            if Config.INPUT_ALL:
                inputs = initial_inputs
            logits.append(logit)
            actions.append(action)
        actions = tf.convert_to_tensor(actions)
        actions = tf.transpose(actions, [1, 0])
        logits = tf.convert_to_tensor(logits)
        logits = tf.transpose(logits, [1, 0, 2])
        pred_final_action = actions
        train_final_action = actions

    with tf.variable_scope("Conv_Critic"):
        out = raw_state
        for i in range(5):
            out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, 2, padding="SAME", activation=tf.nn.relu)
        out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, Config.NUM_OF_CUSTOMERS+1)
        out = tf.reshape(out, [-1, Config.RNN_HIDDEN_DIM])
        base_line_est = tf.layers.dense(tf.layers.dense(out, 10, tf.nn.relu), 1)

    return train_final_action, pred_final_action, base_line_est, logits


def Beam_Search(batch_size, encoder_state, encoder_outputs, train_helper, pred_helper, with_depot_state,
                start_tokens, end_token, keep_prob, raw_state, DECODER_TYPE):
    problem_state = with_depot_state[:, :-1, :]
    beam_width = Config.BEAM_WIDTH
    out_cells = []
    for i in range(Config.LAYERS_STACKED_COUNT):
        with tf.variable_scope('Beam_RNN_{}'.format(i)):
            cell = _build_rnn_cell(keep_prob)
            out_cells.append(cell)
    out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
    if DECODER_TYPE == 0:
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=1)
        out_cell = _build_attention(out_cell, tiled_encoder_outputs, tf.identity)
        out_cell = MaskWrapper(out_cell, DECODER_TYPE)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        # initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))
        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        pred_final_output, pred_final_state, pred_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            pred_decoder, impute_finished=False, maximum_iterations=tf.shape(problem_state)[1])
        pred_final_action = pred_final_output.sample_id
        train_final_output, train_final_state, train_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            train_decoder, impute_finished=False, maximum_iterations=Config.NUM_OF_CUSTOMERS)
        train_final_action = train_final_output.sample_id
        logits = train_final_output.rnn_output
    else:
        train_final_action = tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS])
        logits = tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS, Config.NUM_OF_CUSTOMERS])
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
        out_cell = _build_attention(out_cell, tiled_encoder_outputs, tf.identity)
        out_cell = MaskWrapper(out_cell, DECODER_TYPE)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
        # tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        #     encoder_state, multiplier=beam_width)
        # tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        #     tf.tile([tf.shape(state)[1]], [batch_size]), multiplier=beam_width)
        # pred_initial_state = initial_state.clone(AttnState=pred_initial_state.AttnState.clone(cell_state=tiled_encoder_final_state))
        if Config.STATE_EMBED == 1:
            def beam_embed(sample_ids):
                return(tf.reshape(tf.gather_nd(
                    with_depot_state,
                    tf.concat([tf.reshape(tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, beam_width]), [-1, 1]),
                               tf.reshape(sample_ids, [-1, 1])], 1)), [batch_size, beam_width, Config.RNN_HIDDEN_DIM]))
            pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                out_cell,
                embedding=beam_embed,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
                beam_width=beam_width)
        else:
            def beam_lookup(sample_ids):
                return(tf.reshape(tf.gather_nd(
                    with_depot_state,
                    tf.concat([tf.reshape(tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, beam_width]), [-1, 1]),
                               tf.reshape(sample_ids, [-1, 1])], 1)), [batch_size, beam_width, 2]))
            pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                out_cell,
                embedding=beam_lookup,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
                beam_width=beam_width)
        with tf.variable_scope('decoder'):
            beam_finished, beam_inputs, beam_next_state = pred_decoder.initialize()

            def _shape(batch_size, from_shape):
                if (not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0):
                    return tensor_shape.TensorShape(None)
                else:
                    batch_size = tensor_util.constant_value(
                        ops.convert_to_tensor(
                            batch_size, name="batch_size"))
                    return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

            def _create_ta(s, d):
                return tensor_array_ops.TensorArray(
                    dtype=d,
                    size=0,
                    dynamic_size=True,
                    element_shape=_shape(pred_decoder.batch_size, s))
            beam_outputs_ta = nest.map_structure(_create_ta, pred_decoder.output_size,
                                                 pred_decoder.output_dtype)
            for i in range(Config.NUM_OF_CUSTOMERS):
                beam_outputs, beam_state, beam_next_inputs, beam_finished = pred_decoder.step(i, beam_inputs, beam_next_state)
                beam_next_state = tf.contrib.seq2seq.BeamSearchDecoderState(
                    cell_state=MaskWrapperAttnState(beam_state.cell_state.AttnState,
                                                    mask=beam_state.cell_state.mask +
                                                    tf.one_hot(beam_outputs.predicted_ids,
                                                               depth=Config.NUM_OF_CUSTOMERS, dtype=tf.float32)),
                    finished=beam_state.finished,
                    lengths=beam_state.lengths,
                    log_probs=beam_state.log_probs)
                beam_outputs_ta = nest.map_structure(lambda ta, out: ta.write(i, out), beam_outputs_ta, beam_outputs)
            beam_final_outputs = nest.map_structure(lambda ta: ta.stack(), beam_outputs_ta)
            seq_len = tf.constant(0, shape=[Config.TRAINING_MIN_BATCH_SIZE, Config.BEAM_WIDTH], dtype=tf.int64)
            beam_search_final_outputs, beam_search_final_state = pred_decoder.finalize(
                beam_final_outputs, beam_next_state, sequence_lengths=seq_len)
            pred_final_action = tf.transpose(tf.transpose(beam_search_final_outputs.predicted_ids, [2, 1, 0]), [1, 0, 2])

    with tf.variable_scope("Conv_Critic"):
        out = raw_state
        for i in range(5):
            out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, 2, padding="SAME", activation=tf.nn.relu)
        out = tf.layers.conv1d(out, Config.RNN_HIDDEN_DIM, Config.NUM_OF_CUSTOMERS+1)
        out = tf.reshape(out, [-1, Config.RNN_HIDDEN_DIM])
        base_line_est = tf.layers.dense(tf.layers.dense(out, 10, tf.nn.relu), 1)

    return(train_final_action, pred_final_action, base_line_est, logits)
