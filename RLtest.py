import tensorflow as tf
import numpy as np
from ggplot import *
import pandas as pd
import time

N = 200
batch_size = 50
if N < batch_size:
    batch_size = N
ep_per_batch = 2
total_episodes = 1000000//batch_size//ep_per_batch  # // does integer division
lr = 1e-2


def reward_to_end(s_, actions_):
    t_left = len(s_)
    total = 0
    for i in range(s_.size):
        total += reward(actions_[i], s_[i])
    return(total - t_left)  # the t_left part is subtracting the baseline
    # return(total)


def reward(a_i, s_i):
    if (a_i == 0 and s_i < .5) or (a_i == 1 and s_i >= .5):
        r = 1
    else:
        r = -1
    return(r)


def next_batch(batch_size, ep_per_batch):
    # I tried to set this up in a way to minimize the number of times the ANN has to be evaluated
    s_batch, G_batch, A_batch = [], [], []
    for j in range(ep_per_batch):
        s_ep = np.random.uniform(0, 1, N).reshape(-1, 1)
        prob_weights_ep = sess.run(action, feed_dict={s: s_ep})

        actions = [np.random.binomial(1, p=prob_weights_ep[i][1]) for i in range(N)]
        # actions = np.argmax(prob_weights_ep, axis=1)

        G_batch.append(np.zeros(batch_size))
        idx = np.random.randint(0, N, batch_size)
        for i, t in enumerate(idx):
            G_batch[j][i] = reward_to_end(s_ep[t:], actions[t:])
        A_batch.append(np.asarray(actions)[idx])
        s_batch.append(s_ep[idx])
    return(np.asarray(s_batch).reshape(-1, 1),
           np.asarray(G_batch).reshape(-1),
           np.asarray(A_batch, dtype=np.int32).reshape(-1))


s = tf.placeholder(tf.float32, shape=[None, 1])
reward_holder = tf.placeholder(tf.float32, shape=[None])
action_holder = tf.placeholder(tf.int32, shape=[None])
avg_reward = tf.reduce_mean(reward_holder)

# The way that we talked about setting up the output of the ANN
# output_layer = tf.layers.dense(inputs=s, units=1, activation=tf.sigmoid)
# action = tf.concat([output_layer, 1-output_layer], axis=1)

# Alternative way to set up the output of the ANN
output = tf.layers.dense(inputs=s, units=2, activation=None)
action = tf.nn.softmax(output)

# applies softmax so need to pass unscaled output
neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=action_holder)
loss = tf.reduce_mean(neg_log_prob * reward_holder)
train = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # t_end = time.time() + 60 * 2  # run for 2 min
    # i = 0
    # while time.time() < t_end:
    #     i += 1
    for i in range(total_episodes):
        s_batch, G_batch, A_batch = next_batch(batch_size, ep_per_batch)
        feed_dict = {s: s_batch, reward_holder: G_batch, action_holder: A_batch}
        sess.run(train, feed_dict=feed_dict)
        if i % (total_episodes//20) == 0:
            print("Batch avg_loss = ", sess.run(loss, feed_dict=feed_dict),
                  "\tBatch avg_reward = ", sess.run(avg_reward, feed_dict=feed_dict))

    x = np.linspace(0, 1, 20).reshape(-1, 1)
    probabilities = sess.run(action, feed_dict={s: x})

print(probabilities)
data = pd.DataFrame(np.hstack((x, probabilities)))
data = data.rename(columns={0: "x", 1: "p0", 2: "p1"})
data = pd.melt(data, id_vars=["x"])

p = ggplot(data, aes(x="x", y="value", colour="variable")) + geom_line() + ylim(-.1, 1.1) + labs(title="Episode Length={}".format(N)) + xlab("distance")
p.save(filename="plot_{}_{}_{}_{}.png".format(time.time(), N, lr, total_episodes))
