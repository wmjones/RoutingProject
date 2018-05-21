import tensorflow as tf
import numpy as np
import time

batch_size = 10
MAX_STEPS = 1000000
file_state = np.load('data_state_00.npy', 'r')
file_or_route = np.load('data_or_route_00.npy', 'r')
file_or_cost = np.load('data_or_cost_00.npy', 'r')


def gen_label(state):
    label = np.sum(np.sqrt(np.sum(np.square(state[1:]-state[:-1]), axis=1)))
    return(label)


def cost(raw_state, action):
    costs = []
    for i in range(raw_state.shape[0]):
        state = np.take(raw_state[i], action[i], axis=0)
        cost = np.sum(np.sqrt(np.sum(np.square(state[1:]-state[:-1]), axis=1)))
        cost += np.sum(np.sqrt(np.sum(np.square(state[0], np.array([0, 0], dtype=np.float32)))))
        costs.append(cost)
    costs = np.asarray(costs)
    return(costs.reshape(-1, 1))


def next_batch():
    batch_idx = [np.random.randint(10000) for i in range(batch_size)]
    state_batch = []
    for i in range(batch_size):
        state_i = file_state[batch_idx[i], :]
        # state_i = np.vstack((np.random.rand(20, 2)))
        state_batch.append(state_i)
    state_batch = np.asarray(state_batch)
    route_batch = []
    for i in range(batch_size):
        route_i = file_or_route[batch_idx[i], :]
        route_batch.append(route_i)
    route_batch = np.asarray(route_batch, dtype=np.int32)
    label_batch = []
    for i in range(batch_size):
        label_i = file_or_cost[batch_idx[i]]
        # label_i = gen_label(state_batch[i])
        label_batch.append(label_i)
    label_batch = np.asarray(label_batch).reshape(-1, 1)
    return(state_batch, label_batch, route_batch)


problem_state_with_depot = tf.placeholder(tf.float32, shape=[None, 20, 2])
problem_label = tf.placeholder(tf.float32, shape=[None, 1])
sampled_cost = tf.placeholder(tf.float32, shape=[None, 1])
problem_action = tf.placeholder(tf.int32, shape=[None, 20])
problem_state = problem_state_with_depot[:, :-1, :]
for i in range(5):
    problem_state = tf.layers.conv1d(problem_state, 128, 1, padding="SAME", activation=tf.nn.relu)
initial_inputs = tf.zeros([batch_size, 128])

with tf.variable_scope("Actor"):
    cell = tf.nn.rnn_cell.LSTMCell(128)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=128, memory=problem_state,
                                                            probability_fn=tf.identity)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, output_attention=True)
    state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    mask = tf.zeros([batch_size, 19], dtype=tf.float32)
    actions = []
    logits = []
    inputs = initial_inputs
    for i in range(19):
        outputs, state = cell(inputs, state)
        logit = state.alignments-mask*1e6
        action = tf.argmax(logit, axis=1, output_type=tf.int32)
        logit = 10*tf.nn.tanh(logit)
        mask = mask + tf.one_hot(action, 19, dtype=tf.float32)
        inputs = tf.gather_nd(problem_state, tf.concat([tf.reshape(tf.range(0, batch_size), [-1, 1]),
                                                        tf.reshape(action, [-1, 1])], 1))
        logits.append(logit)
        actions.append(action)
    actions = tf.convert_to_tensor(actions)
    actions = tf.transpose(actions, [1, 0])
    logits = tf.convert_to_tensor(logits)
    logits = tf.transpose(logits, [1, 0, 2])
    pred_final_action = actions
    train_final_action = actions

# with tf.variable_scope("Conv_Critic"):
#     out = problem_state
#     for i in range(5):
#         out = tf.layers.conv1d(out, 128, 2, padding="SAME", activation=tf.nn.relu)
#     out = tf.reshape(out, [-1, 128*20])
#     # out = tf.layers.conv1d(out, 128, 20, activation=tf.nn.relu)
#     # out = tf.reshape(out, [-1, 128])
#     base_line_est = tf.layers.dense(tf.layers.dense(out, 10, tf.nn.relu), 1)

# loss = tf.losses.mean_squared_error(problem_label, base_line_est)

# weights = tf.to_float(tf.tile(tf.reshape(tf.range(
#     1, tf.divide(1, tf.shape(state)[1]), -tf.divide(1, tf.shape(problem_state)[1])),
#                                               [1, -1]), [batch_size, 1]))
loss = tf.contrib.seq2seq.sequence_loss(
    logits=logits,
    targets=problem_action[:, :-1],
    weights=tf.ones([batch_size, tf.shape(problem_state)[1]])
    # weights=weights
)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.name_scope("Loss"):
    tf.summary.scalar("Loss", loss)
with tf.name_scope("Performace"):
    tf.summary.scalar("Avg_sampled_cost", tf.reduce_mean(sampled_cost))

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
log_name = "./logs/" + "test_" + str(time.time())
log_writer = tf.summary.FileWriter(log_name)
merged = tf.summary.merge_all()
with tf.Session(config=config) as sess:
    log_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(MAX_STEPS):
        new_state, new_label, new_route = next_batch()
        feed_dict = {problem_state_with_depot: new_state, problem_label: new_label, problem_action: new_route}
        if i % 1000 == 0:
            _, batch_loss, pred_route = sess.run([train_op, loss, actions], feed_dict=feed_dict)
            new_cost = cost(new_state, pred_route)
            feed_dict = {problem_state_with_depot: new_state, problem_label: new_label, problem_action: new_route,
                         sampled_cost: new_cost}
            summary, _ = sess.run([merged, loss], feed_dict=feed_dict)
            log_writer.add_summary(summary, i)
            print("step: " + str(i) + "    loss: " + str(batch_loss))
            if i % 1000 == 0:
                print("pred_route:")
                print(str(pred_route))
                print("or_route:")
                print(str(new_route))
                print("pred_cost:")
                print(cost(new_state, pred_route))
                print("or_cost:")
                print(new_label)
                print()
        else:
            sess.run(train_op, feed_dict=feed_dict)
    # print(feed_dict)
    # print(sess.run(base_line_est, feed_dict=feed_dict))
log_writer.close()
