import tensorflow as tf
# import time
from MaskWrapper import MaskWrapper
from MaskWrapper import MaskWrapperAttnState
from Model import Encoder, Helper, Decoder, Reza_Model
# from MaskWrapper import MaskWrapperState
import numpy as np

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
            tf.summary.scalar("REINFORCE", Config.REINFORCE)

        if Config.DIRECTION == 8:
            self.state = tf.layers.conv1d(self.state, Config.RNN_HIDDEN_DIM, 1)
        self.enc_inputs = self.state

        # ENCODER
        if Config.DIRECTION != 9:
            self.encoder_outputs, self.encoder_state = Encoder(self.enc_inputs)
        if Config.DIRECTION == 8:
            self.encoder_outputs = self.state

        # HELPERS
        self.gather_ids = tf.concat([tf.expand_dims(
            tf.reshape(tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), [1, tf.shape(self.state)[1]]), [-1]), -1),
                                     tf.reshape(self.or_action, [-1, 1])], -1)
        self.training_inputs = tf.reshape(tf.gather_nd(self.state, self.gather_ids), [self.batch_size, tf.shape(self.state)[1], 2])
        self.training_inputs = tf.concat([tf.zeros([self.batch_size, 1, 2]), self.training_inputs], axis=1)
        self.training_inputs = self.training_inputs[:, :-1, :]
        train_helper, pred_helper = Helper(self.state, self.batch_size, self.training_inputs, self.start_tokens, self.end_token)

        # DECODER
        if Config.DIRECTION != 9:
            train_decoder, pred_decoder, critic_decoder = Decoder(self.batch_size, self.encoder_state, self.encoder_outputs,
                                                                  train_helper, pred_helper, self.state, self.start_tokens,
                                                                  self.end_token)
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
                self.base_line_est = tf.layers.dense(self.critic_final_state.cell_state.c,
                                                     Config.RNN_HIDDEN_DIM, activation=tf.nn.relu)
            self.base_line_est = tf.layers.dense(self.base_line_est, 1)
        else:
            self.train_final_action, self.pred_final_action, self.base_line_est, self.logits = Reza_Model(self.batch_size, self.state)

        self.critic_loss = tf.losses.mean_squared_error(self.sampled_cost, self.base_line_est)

        tf.summary.histogram("LocationStartDist", tf.transpose(self.pred_final_action, [1, 0])[0])
        tf.summary.histogram("LocationEndDist", tf.transpose(self.pred_final_action, [1, 0])[-1])

        self.weights = tf.ones([self.batch_size, tf.shape(self.state)[1]])

        if Config.DIRECTION != 9:
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
            self.neg_log_prob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=self.train_final_action)
            R = tf.stop_gradient(self.sampled_cost)
            V = tf.stop_gradient(self.base_line_est)
            # V = tf.constant(4.0)
            self.loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(self.neg_log_prob, axis=1), R-V))

        with tf.name_scope("train"):
            if Config.GPU == 1:
                colocate = True
            else:
                colocate = False
            self.lr = tf.train.exponential_decay(
                Config.LEARNING_RATE, self.global_step, 10000,
                .9, staircase=True, name="learning_rate")
            self.critic_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)
            if Config.MAX_GRAD != 0:
                self.params = tf.trainable_variables()
                self.gradients = tf.gradients(self.loss, self.params, colocate_gradients_with_ops=colocate)
                self.clipped_gradients, gradient_norm = tf.clip_by_global_norm(self.gradients, Config.MAX_GRAD)
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.apply_gradients(zip(self.clipped_gradients, self.params), global_step=self.global_step)
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
            tf.summary.scalar("Avg_or_cost", tf.reduce_mean(self.or_cost))
            tf.summary.scalar("Avg_sampled_cost", tf.reduce_mean(self.sampled_cost))
        self.base_line_est = tf.zeros(shape=[self.batch_size, 1])

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
        # pred_act, logits = self.sess.run(
        #     [self.pred_final_action, self.logits],
        #     feed_dict=feed_dict)
        # print("actions")
        # print(pred_act)
        # print("logits")
        # print(logits)
        # print("train_action")
        # print(train_act)
        # print("softmax logits")
        # print(softmax_log)
        # print("neg_log_prob")
        # print(neg_log_prob)
        # print("sum")
        # print(total_sum)
        # print("rnn_output")
        # print(train_logits)
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
            self.sess.run([self.train_op, self.critic_train_op], feed_dict=feed_dict)
        if step % 1000 == 0 and Config.TRAIN:
            print("Saving Model...")
            self._model_save()
            print(self.sess.run([self.pred_final_action], feed_dict=feed_dict))
            print(self.sess.run([self.or_action], feed_dict=feed_dict))

    def _create_tensor_board(self):
        # for added metadata https://www.tensorflow.org/programmers_guide/graph_viz
        log_name = str(Config.PATH) + "logs/" + self.name
        self.log_writer = tf.summary.FileWriter(log_name)
        self.log_writer.add_graph(self.sess.graph)
        self.merged = tf.summary.merge_all()

    def finish(self):
        if Config.TRAIN == 1:
            self._model_save()
        self.log_writer.close()
