from multiprocessing import Process, Queue, Value
import numpy as np
import time
import threading

from Config import Config
from Environment import Environment
from OR_Tool import OR_Tool
# from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q

        self.env = Environment()
        # self.num_actions = self.env.get_num_actions()
        # self.actions = np.arange(self.num_actions)
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    # https://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
    # @staticmethod
    # def _accumulate_rewards(experiences, discount_factor=.99, terminal_reward=0):
    #     reward_sum = terminal_reward
    #     for t in reversed(range(0, len(experiences)-1)):
    #         r = experiences[t].reward
    #         reward_sum = discount_factor*reward_sum + r
    #         experiences[t].reward = reward_sum
    #     return experiences[:-1]

    # def convert_data(self, experiences):
    #     x_ = np.array([exp.state for exp in experiences])
    #     a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
    #     r_ = np.array([exp.reward for exp in experiences])
    #     return x_, r_, a_

    # def predict(self, state):
    #     self.prediction_q.put((self.id, state))
    #     p, v = self.wait_q.get()  # how does it know to get the right one?
    #     return p, v

    # def select_action(self, prediction):
    #     if Config.PLAY_MODE:
    #         action = np.argmax(prediction)
    #     else:
    #         action = np.random.choice(self.actions, p=prediction)
    #     return action

    # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    # def run_episode(self):
    #     self.env.reset()
    #     done = False
    #     experiences = []
    #     time_count = 0
    #     reward_sum = 0.0

    #     while not done:
    #         if self.env.current_state is None:  # not sure what is going on
    #             self.env.step(0)
    #             continue        # why

    #         prediction, value = self.predict(self.env.current_state)
    #         action = self.select_action(prediction)
    #         reward, done = self.env.step(action)
    #         reward_sum += reward
    #         exp = Experience(self.env.previous_state, action, prediction, reward, done)
    #         experiences.append(exp)

    #         if done or time_count == Config.TIME_MAX:
    #             terminal_reward = 0 if done else value  # does it being value cause any problems?

    #             updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
    #             x_, r_, a_ = self.convert_data(updated_exps)  # what is this doing?
    #             yield x_, r_, a_, reward_sum

    #             # surprised code reaches here since it is after yield
    #             time_count = 0
    #             experiences = [experiences[-1]]
    #             reward_sum = 0.0

    #         time_count += 1

    def predict(self, state):
        self.prediction_q.put((self.id, state))
        a, v = self.wait_q.get()
        return a, v

    def run_episode(self):
        self.env.reset()
        action, base_line = self.predict(self.env.current_state)
        sampled_value = self.env.G(action)
        if Config.OR_TOOLS:
            or_model = OR_Tool(self.env.current_state)
            or_model.solve()
            # add in what i want to return from this and where the data should go (maybe to gpu to help train)
        return action, base_line, sampled_value

    def run(self):
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            a_, b_, r_ = self.run_episode()
            x_ = self.env.current_state
            self.training_q.put(([x_], [a_], [r_]))
