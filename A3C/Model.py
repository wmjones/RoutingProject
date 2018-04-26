import tensorflow as tf
from Config import Config
from MyWrapper import MaskWrapper, MaskWrapperAttnState
from tensorflow.python.ops.distributions import categorical


def _build_rnn_cell(keep_prob):
    # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell


def _build_attention(cell, memory):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=Config.RNN_HIDDEN_DIM, memory=memory)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, output_attention=True)
    # attn_cell = AttentionWrapper(cell, attention_mechanism, output_attention=True)
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
            # in_cell = tf.contrib.rnn.BasicLSTMCell(Config.RNN_HIDDEN_DIM)
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
                    # cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
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
        if Config.DIRECTION == 4:
            in_cells = []
            for i in range(Config.LAYERS_STACKED_COUNT):
                with tf.variable_scope('ENC_RNN_{}'.format(i)):
                    # cell = tf.nn.rnn_cell.LSTMCell(Config.RNN_HIDDEN_DIM)
                    cell = _build_rnn_cell(keep_prob)
                    in_cells.append(cell)
            in_cell = tf.nn.rnn_cell.MultiRNNCell(in_cells)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(in_cell,
                                                               enc_inputs,
                                                               dtype=tf.float32)
        return encoder_outputs, encoder_state


def Helper(problem_state, batch_size, training_inputs, start_tokens, end_token):
    with tf.variable_scope("Helper"):
        if Config.REINFORCE == 1:
            if Config.STOCHASTIC == 1:
                def initialize_fn():
                    if Config.STATE_EMBED == 0:
                        return (tf.tile([False], [batch_size]), tf.zeros([batch_size, 2]))
                    else:
                        return (tf.tile([False], [batch_size]), tf.zeros([batch_size, Config.RNN_HIDDEN_DIM]))

                def sample_fn(time, outputs, state):
                    logits = outputs / Config.SOFTMAX_TEMP
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
                    # finished = tf.Print(finished, [next_state.mask], summarize=1000)
                    return (finished, next_inputs, next_state)
                train_helper = tf.contrib.seq2seq.CustomHelper(
                    initialize_fn=initialize_fn,
                    sample_fn=sample_fn,
                    next_inputs_fn=next_inputs_fn
                )
                pred_helper = train_helper
            else:
                pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        lambda sample_ids: tf.gather_nd(
                            problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                                      tf.reshape(sample_ids, [-1, 1])], 1)
                        ),
                        start_tokens,
                        end_token)
                train_helper = pred_helper
        else:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda sample_ids: tf.gather_nd(
                    problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                              tf.reshape(sample_ids, [-1, 1])], 1)
                ),
                start_tokens,
                end_token)
            train_helper = tf.contrib.seq2seq.TrainingHelper(training_inputs,
                                                             tf.fill([batch_size], Config.NUM_OF_CUSTOMERS))
        return train_helper, pred_helper


def Decoder(batch_size, encoder_state, encoder_outputs, train_helper, pred_helper, problem_state, start_tokens, end_token, keep_prob):
    # with tf.variable_scope("Decoder"):
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

        critic_out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('CRITIC_RNN_{}'.format(i)):
                cell = _build_rnn_cell(keep_prob)
                critic_out_cells.append(cell)
        critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        critic_out_cell = MaskWrapper(critic_out_cell)
        critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))
        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    if Config.DIRECTION == 2:
        out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
        out_cell = _build_attention(out_cell, encoder_outputs)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell, Config.NUM_OF_CUSTOMERS)
        out_cell = MaskWrapper(out_cell)
        initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(AttnState=initial_state.AttnState.clone(cell_state=encoder_state))

        critic_out_cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
        critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        critic_out_cell = MaskWrapper(critic_out_cell)
        critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
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

        critic_out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('CRITIC_DEC_RNN_{}'.format(i)):
                cell = tf.contrib.rnn.BasicLSTMCell(2*Config.RNN_HIDDEN_DIM)
                critic_out_cells.append(cell)
        critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        critic_out_cell = MaskWrapper(critic_out_cell)
        critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        critic_initial_state = critic_initial_state.clone(AttnState=critic_initial_state.AttnState.clone(cell_state=encoder_state))

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    if Config.DIRECTION == 4:
        critic_out_cells = [_build_rnn_cell(keep_prob)]
        critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        critic_out_cell = MaskWrapper(critic_out_cell)
        critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        critic_initial_state = critic_initial_state.clone(cell_state=encoder_state)
        critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
        out_cells = []
        for i in range(Config.LAYERS_STACKED_COUNT):
            with tf.variable_scope('RNN_{}'.format(i)):
                cell = _build_rnn_cell(keep_prob)
                out_cells.append(cell)
        out_cell = tf.nn.rnn_cell.MultiRNNCell(out_cells)
        with tf.variable_scope("beam"):
            train_out_cell = _build_attention(out_cell, encoder_outputs)
            train_out_cell = tf.contrib.rnn.OutputProjectionWrapper(train_out_cell, Config.NUM_OF_CUSTOMERS)
            train_out_cell = MaskWrapper(train_out_cell)
            initial_state = train_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            initial_state = initial_state.clone(cell_state=encoder_state)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(train_out_cell, train_helper, initial_state)
        with tf.variable_scope("beam", reuse=True):
            beam_width = 10
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=beam_width)
            # tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            #     tf.tile([tf.shape(state)[1]], [batch_size]), multiplier=beam_width)
            pred_out_cell = _build_attention(out_cell, tiled_encoder_outputs)
            pred_out_cell = tf.contrib.rnn.OutputProjectionWrapper(pred_out_cell, Config.NUM_OF_CUSTOMERS)
            pred_out_cell = MaskWrapper(pred_out_cell)
            pred_initial_state = pred_out_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size * beam_width)
            pred_initial_state = pred_initial_state.clone(
                cell_state=tiled_encoder_final_state)
            if Config.DEC_EMB == 1:
                def beam_embed(x):
                    return(
                        tf.reshape(tf.nn.conv1d(tf.reshape(tf.gather_nd(problem_state, tf.concat(
                            [tf.reshape(tf.range(batch_size), [-1, 1]),
                             tf.reshape(x, [-1, 1])], -1)), [batch_size, 1, 2]), W_embed, 1, "VALID"),
                                   [batch_size, Config.RNN_HIDDEN_DIM])
                    )
                # even with logit pen it is outputing infeas routes
                pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    pred_out_cell,
                    embedding=lambda ids: tf.transpose(
                        tf.map_fn(beam_embed, tf.transpose(ids, [1, 0]), dtype=tf.float32),
                        [1, 0, 2]),
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=pred_initial_state,
                    beam_width=beam_width)
            else:
                pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    pred_out_cell,
                    embedding=lambda sample_ids: tf.transpose(
                        tf.reshape(tf.gather_nd(
                            problem_state,
                            tf.concat([tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [beam_width, 1]),
                                       tf.reshape(sample_ids, [-1, 1])], 1)), [beam_width, batch_size, 2]), [1, 0, 2]),
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=pred_initial_state,
                    beam_width=beam_width)
            # pred_decoder = tf.contrib.seq2seq.BasicDecoder(train_out_cell, pred_helper, initial_state)
    if Config.DIRECTION == 5:
        cell = tf.contrib.grid_rnn.Grid2LSTMCell(Config.RNN_HIDDEN_DIM,
                                                 use_peepholes=True,
                                                 output_is_tuple=False,
                                                 state_is_tuple=False)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
        cell = MaskWrapper(cell, cell_is_attention=False)
        initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state)
    if Config.DIRECTION == 6:
        cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
                                                 use_peepholes=True,
                                                 output_is_tuple=False,
                                                 state_is_tuple=False)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * Config.LAYERS_STACKED_COUNT, state_is_tuple=False)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
        cell = MaskWrapper(cell, cell_is_attention=False)
        initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state)
    if Config.DIRECTION == 7:
        cell = tf.contrib.grid_rnn.Grid3LSTMCell(Config.RNN_HIDDEN_DIM,
                                                 use_peepholes=True,
                                                 output_is_tuple=False,
                                                 state_is_tuple=False)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, Config.NUM_OF_CUSTOMERS)
        cell = MaskWrapper(cell, cell_is_attention=False)
        initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, initial_state,
                                                        output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, initial_state,
                                                       output_layer=tf.layers.Dense(Config.NUM_OF_CUSTOMERS+1))
    if Config.DIRECTION == 8:
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

        critic_out_cells = [_build_rnn_cell(keep_prob)]
        critic_out_cell = tf.nn.rnn_cell.MultiRNNCell(critic_out_cells)
        critic_out_cell = _build_attention(critic_out_cell, encoder_outputs)
        critic_out_cell = tf.contrib.rnn.OutputProjectionWrapper(critic_out_cell, Config.NUM_OF_CUSTOMERS)
        critic_out_cell = MaskWrapper(critic_out_cell)
        critic_initial_state = critic_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        train_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, train_helper, initial_state)
        pred_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, pred_helper, initial_state)
        critic_decoder = tf.contrib.seq2seq.BasicDecoder(critic_out_cell, pred_helper, critic_initial_state)
    return train_decoder, pred_decoder, critic_decoder


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
