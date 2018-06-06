import tensorflow as tf
# import time
# from MaskWrapper import MaskWrapper
# from MaskWrapper import MaskWrapperAttnState
from Model import Encoder, Helper, Decoder, Reza_Model, Wyatt_Model, Beam_Search
# from MaskWrapper import MaskWrapperState
import numpy as np
# from Environment import Environment
from Config import Config


class NetworkVP:
    def __init__(self, device, DECODER_TYPE):
        if DECODER_TYPE == 0:
            self.graph = tf.Graph()
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
            with self.graph.as_default():
                self._create_graph(DECODER_TYPE=DECODER_TYPE)
                self.saver = tf.train.Saver()
                self.sess.run(tf.global_variables_initializer())
                self.name = Config.MODEL_NAME
                print("Running Model ", self.name)
                if Config.RESTORE == 1:
                    print("Restoring Parameters from latest checkpoint:")
                    latest_checkpoint = tf.train.latest_checkpoint(str(Config.PATH) + 'checkpoint/' + Config.MODEL_NAME + '/')
                    print(latest_checkpoint)
                    self.saver.restore(self.sess, latest_checkpoint)
                    self.name = self.name + 'res'
                    self._create_tensor_board()
                else:
                    self._create_tensor_board()
        else:
            self.graph = tf.Graph()
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
            self.sess = tf.Session(
                graph=self.graph,
                config=config
            )
            with self.graph.as_default():
                if Config.SAMPLING == 1:
                    Config.STOCHASTIC = 1
                self._create_graph(DECODER_TYPE=DECODER_TYPE)
                self.saver = tf.train.Saver()
                self.sess.run(tf.global_variables_initializer())
                self.name = Config.MODEL_NAME
                latest_checkpoint = tf.train.latest_checkpoint(str(Config.PATH) + 'checkpoint/' + Config.MODEL_NAME + '/')
                # print(latest_checkpoint)
                self.saver.restore(self.sess, latest_checkpoint)
                # self.saver.save(self.sess, str(Config.PATH) + 'checkpoint/' + self.name + '_beam' + '/model.ckpt')

    def _model_save(self):
        self.saver.save(self.sess, str(Config.PATH) + 'checkpoint/' + self.name + '/model.ckpt')

    def _create_tensor_board(self):
        # for added metadata https://www.tensorflow.org/programmers_guide/graph_viz
        log_name = str(Config.PATH) + "logs/" + self.name
        self.log_writer = tf.summary.FileWriter(log_name)
        self.log_writer.add_graph(self.sess.graph)
        self.merged = tf.summary.merge_all()

    def finish(self):
        self._model_save()
        self.log_writer.close()

    def _create_graph(self, DECODER_TYPE):
        self.raw_state = tf.placeholder(tf.float32, shape=[None, Config.NUM_OF_CUSTOMERS+1, 2], name='State')
        # self.current_location = tf.placeholder(tf.float32, shape=[None, 2], name='Current_Location')
        self.current_location = self.raw_state[:, -1]
        self.sampled_cost = tf.placeholder(tf.float32, [None, 1], name='Sampled_Cost')
        self.batch_size = tf.shape(self.raw_state)[0]
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.input_lengths = tf.convert_to_tensor([Config.NUM_OF_CUSTOMERS]*(self.batch_size))
        self.or_route = tf.placeholder(tf.int32, shape=[None, Config.NUM_OF_CUSTOMERS+1])
        # self.last_pred_route = tf.placeholder(tf.int32, shape=[None, Config.BEAM_WIDTH, Config.NUM_OF_CUSTOMERS])
        self.or_cost = tf.placeholder(tf.float32, shape=[None, 1])
        self.difference_in_length = tf.reduce_mean(self.sampled_cost - self.or_cost)
        self.relative_length = tf.reduce_mean(self.sampled_cost/self.or_cost)
        self.start_tokens = tf.placeholder(tf.int32, shape=[None])
        self.end_token = -1
        self.MA_baseline = tf.Variable(7.0, dtype=tf.float32, trainable=False)

        if Config.STATE_EMBED == 1:
            self.with_depot_state = self.raw_state
            for i in range(2):
                self.with_depot_state = tf.layers.conv1d(self.with_depot_state, Config.RNN_HIDDEN_DIM, 1,
                                                         padding="SAME", activation=tf.nn.relu)
            self.with_depot_state = tf.layers.conv1d(self.with_depot_state, Config.RNN_HIDDEN_DIM, 1,
                                                     padding="VALID")
        else:
            self.with_depot_state = self.raw_state
        self.state = self.with_depot_state[:, :-1, :]

        # ENCODER
        if Config.DIRECTION == 4 or Config.DIRECTION == 5 or Config.DIRECTION == 6:
            self.encoder_outputs = self.state
            self.encoder_state = None
        if Config.DIRECTION < 6 and Config.DIRECTION != 4 and Config.DIRECTION != 5 and Config.DIRECTION != 6:
            self.encoder_outputs, self.encoder_state = Encoder(self.state, self.keep_prob)

        # HELPERS
        self.training_index = tf.concat([tf.expand_dims(self.start_tokens, -1), self.or_route], axis=1)
        self.training_index = self.training_index[:, :-1]
        self.gather_ids = tf.concat([tf.expand_dims(
            tf.reshape(tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), [1, tf.shape(self.with_depot_state)[1]]), [-1]), -1),
                                     tf.reshape(self.training_index, [-1, 1])], -1)
        if Config.STATE_EMBED == 0:
            self.training_inputs = tf.reshape(tf.gather_nd(self.with_depot_state, self.gather_ids),
                                              [self.batch_size, tf.shape(self.with_depot_state)[1], 2])
        else:
            self.training_inputs = tf.reshape(tf.gather_nd(self.with_depot_state, self.gather_ids),
                                              [self.batch_size, tf.shape(self.with_depot_state)[1], Config.RNN_HIDDEN_DIM])
        train_helper, pred_helper = Helper(self.with_depot_state, self.batch_size, self.training_inputs,
                                           self.start_tokens, self.end_token)

        # DECODER
        if Config.DIRECTION < 6:
            train_decoder, pred_decoder, critic_network_pred = Decoder(self.batch_size, self.encoder_state, self.encoder_outputs,
                                                                       train_helper, pred_helper, self.state, self.start_tokens,
                                                                       self.end_token, self.keep_prob, self.raw_state, DECODER_TYPE)

            self.train_final_output, self.train_final_state, train_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                train_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
            self.train_final_action = self.train_final_output.sample_id

            self.pred_final_output, self.pred_final_state, pred_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                pred_decoder, impute_finished=False, maximum_iterations=tf.shape(self.state)[1])
            self.pred_final_action = self.pred_final_output.sample_id
            self.base_line_est = critic_network_pred

        if Config.DIRECTION == 6:
            self.train_final_action, self.pred_final_action, self.base_line_est, self.logits = Beam_Search(
                self.batch_size, self.encoder_state, self.encoder_outputs,
                train_helper, pred_helper, self.with_depot_state, self.start_tokens,
                self.end_token, self.keep_prob, self.raw_state, DECODER_TYPE)

        if Config.DIRECTION == 9:
            self.train_final_action, self.pred_final_action, self.base_line_est, self.logits = Reza_Model(self.batch_size,
                                                                                                          self.with_depot_state)
        if Config.DIRECTION == 10:
            self.train_final_action, self.pred_final_action, self.base_line_est, self.logits = Wyatt_Model(self.batch_size,
                                                                                                           self.state,
                                                                                                           self.raw_state)

        if DECODER_TYPE == 0:
            # x = tf.range(0, 19, dtype=tf.int32)
            # x = [tf.random_shuffle(x)]

            # for i in range(499):
            #     y = tf.range(0, 19, dtype=tf.int32)
            #     y = [tf.random_shuffle(y)]
            #     x = tf.concat((x, y), axis=0)
            # self.pred_final_action = x[:self.batch_size, :]

            self.critic_loss = tf.losses.mean_squared_error(self.sampled_cost, self.base_line_est)

            if Config.DIRECTION < 6:
                self.logits = self.train_final_output.rnn_output

            if Config.LOGIT_CLIP_SCALAR != 0:
                self.logits = Config.LOGIT_CLIP_SCALAR*tf.nn.tanh(self.logits)

            if Config.REINFORCE == 0:
                # self.weights = tf.to_float(tf.tile(tf.reshape(tf.range(
                #     1, tf.divide(1, tf.shape(self.state)[1]), -tf.divide(1, tf.shape(self.state)[1])),
                #                                               [1, -1]), [self.batch_size, 1]))
                self.actor_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.logits,
                    targets=self.or_route[:, :-1],
                    weights=tf.ones([self.batch_size, tf.shape(self.state)[1]])
                    # weights=self.weights
                )
            else:
                self.neg_log_prob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                      labels=self.train_final_action)
                self.R = tf.stop_gradient(self.sampled_cost)
                if Config.MOVING_AVERAGE == 1:
                    assign = tf.assign(self.MA_baseline, self.MA_baseline*.99 + tf.reduce_mean(self.R)*.01)
                    with tf.control_dependencies([assign]):
                        V = self.MA_baseline
                        self.actor_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.neg_log_prob, axis=1), self.R-V))
                elif Config.USE_OR_COST == 1:
                    V = tf.stop_gradient(self.or_cost)
                    self.actor_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.neg_log_prob, axis=1), (self.R-V)/5))
                else:
                    V = tf.stop_gradient(self.base_line_est)
                    self.actor_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.neg_log_prob, axis=1), self.R-V))

            with tf.name_scope("Train"):
                if Config.GPU == 1:
                    colocate = True
                else:
                    colocate = False
                if Config.LR_DECAY_OFF == 0:
                    self.lr = tf.train.exponential_decay(
                        Config.LEARNING_RATE, self.global_step, 100000,
                        .9, staircase=True, name="learning_rate")
                else:
                    self.lr = Config.LEARNING_RATE
                self.train_critic_op = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)
                if Config.MAX_GRAD != 0:
                    self.params = tf.trainable_variables()
                    self.gradients = tf.gradients(self.actor_loss, self.params, colocate_gradients_with_ops=colocate)
                    opt = tf.train.AdamOptimizer(self.lr)
                    self.clipped_gradients, gradient_norm = tf.clip_by_global_norm(self.gradients, Config.MAX_GRAD)
                    self.train_actor_op = opt.apply_gradients(zip(self.clipped_gradients, self.params), global_step=self.global_step)
                    tf.summary.scalar("grad_norm", gradient_norm)
                    tf.summary.scalar("LearningRate", self.lr)
                else:
                    self.train_actor_op = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss,
                                                                                   global_step=self.global_step,
                                                                                   colocate_gradients_with_ops=colocate)
            # # for gradient clipping https://github.com/tensorflow/nmt/blob/master/nmt/model.py

            with tf.name_scope("Loss"):
                tf.summary.scalar("Loss", self.actor_loss)
                tf.summary.scalar("Critic_Loss", self.critic_loss)
            with tf.name_scope("Performace"):
                tf.summary.scalar("Relative Critic Loss", tf.reduce_mean(self.base_line_est/self.or_cost))
                tf.summary.scalar("Relative Critic Loss to Sampled", tf.reduce_mean(self.base_line_est/self.sampled_cost))
                tf.summary.scalar("difference_in_length", self.difference_in_length)
                tf.summary.scalar("relative_length", self.relative_length)
                tf.summary.scalar("Avg_or_cost", tf.reduce_mean(self.or_cost))
                tf.summary.scalar("Avg_sampled_cost", tf.reduce_mean(self.sampled_cost))
                # tf.summary.histogram("LocationStartDist", tf.transpose(self.pred_final_action, [1, 0])[0])
                # tf.summary.histogram("LocationEndDist", tf.transpose(self.pred_final_action, [1, 0])[-1])
            with tf.name_scope("Config"):
                tf.summary.scalar("REINFORCE", Config.REINFORCE)
                tf.summary.scalar("DIRECTION", Config.DIRECTION)
                tf.summary.scalar("NUM_OF_CUSTOMERS", Config.NUM_OF_CUSTOMERS)
                tf.summary.scalar("StateEmbed", tf.cast(Config.STATE_EMBED, tf.int32))
                tf.summary.scalar("MAX_GRAD", Config.MAX_GRAD)
                tf.summary.scalar("LogitPen", Config.LOGIT_PENALTY)
                tf.summary.scalar("batch_size", self.batch_size)
                tf.summary.scalar("Config.LAYERS_STACKED_COUNT", Config.LAYERS_STACKED_COUNT)
                tf.summary.scalar("RNN_HIDDEN_DIM", Config.RNN_HIDDEN_DIM)
                tf.summary.scalar("RUN_TIME", Config.RUN_TIME)
                tf.summary.scalar("LOGIT_CLIP_SCALAR", Config.LOGIT_CLIP_SCALAR)
                tf.summary.scalar("EncEmb", tf.cast(Config.ENC_EMB, tf.int32))
                tf.summary.scalar("DecEmb", tf.cast(Config.DEC_EMB, tf.int32))
                tf.summary.scalar("Droput", tf.cast(Config.DROPOUT, tf.int32))
                tf.summary.scalar("GPU", Config.GPU)

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict(self, state, depot_idx, num_samples=1):
        feed_dict = {self.raw_state: state, self.start_tokens: depot_idx}
        sampled_pred_route, sampled_pred_cost = self.sess.run([self.pred_final_action, self.base_line_est], feed_dict=feed_dict)
        pred_batch_size = sampled_pred_route.shape[0]
        pred_route = []
        pred_cost = []
        for i in range(pred_batch_size):
            pred_route.append([sampled_pred_route[i]])
            pred_cost.append([sampled_pred_cost[i]])
        for i in range(num_samples-1):
            sampled_pred_route_i, sampled_pred_cost_i = self.sess.run(
                [self.pred_final_action, self.base_line_est], feed_dict=feed_dict)
            for j in range(pred_batch_size):
                pred_route[j] = np.vstack((pred_route[j], sampled_pred_route_i[j]))
                pred_cost[j] = np.vstack((pred_cost[j], sampled_pred_cost_i[j]))
        pred_route = np.asarray(pred_route)
        pred_cost = np.asarray(pred_cost)
        return pred_route, pred_cost

    def train(self, state, depot_location, or_action=0, sampled_cost=0, or_cost=0):
        if Config.REINFORCE == 0:
            feed_dict = {self.raw_state: state, self.or_route: or_action,
                         self.start_tokens: depot_location, self.keep_prob: 1}
            self.sess.run([self.train_actor_op], feed_dict=feed_dict)
        else:
            feed_dict = {self.raw_state: state, self.sampled_cost: sampled_cost,
                         self.start_tokens: depot_location, self.keep_prob: 1, self.or_cost: or_cost}
            self.sess.run([self.train_actor_op, self.train_critic_op], feed_dict=feed_dict)

    def summary(self, state, or_cost, or_route, depot_location, pred_cost, sampled_cost):
        step = self.get_global_step()
        feed_dict = {self.raw_state: state, self.or_cost: or_cost, self.or_route: or_route,
                     self.start_tokens: depot_location,
                     self.sampled_cost: sampled_cost}
        _, _, summary, loss, diff, pred_route, train_route = self.sess.run([
            self.train_actor_op, self.train_critic_op, self.merged,
            self.actor_loss, self.relative_length, self.pred_final_action,
            self.train_final_action], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)
        print("step_loss_diff:")
        print(step, loss, diff)
        print()

    # def beam_search_evaluation(self, state, depot_idx):
    #     latest_checkpoint = tf.train.latest_checkpoint(str(Config.PATH) + 'checkpoint/' + Config.MODEL_NAME + '/')
    #     self.saver.restore(self.sess, latest_checkpoint)
    #     feed_dict = {self.raw_state: state, self.start_tokens: depot_idx}
    #     pred_route, pred_cost = self.sess.run([self.pred_final_action, self.base_line_est], feed_dict=feed_dict)
    #     return pred_route, pred_cost
