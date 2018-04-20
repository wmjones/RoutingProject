from multiprocessing import Process, Queue, Value
import numpy as np
import time
# import threading

from Config import Config
from Environment import Environment
from OR_Tool import OR_Tool


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q

        self.env = Environment()
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    def predict(self, state, current_location, depot_idx):
        self.prediction_q.put((self.id, state, current_location, depot_idx))
        a, v = self.wait_q.get()
        return a, v

    def run_episode(self):
        if Config.FROM_FILE == 1:
            file_state = np.load('data_state.npy', 'r')
            file_or_route = np.load('data_or_route.npy', 'r')
            file_or_cost = np.load('data_or_cost.npy', 'r')
            self.batch_idx = np.random.randint(3)
            self.env.current_state = file_state[self.batch_idx, :]
            self.env.distance_matrix = self.env.get_distance_matrix()
            current_location = self.env.get_current_location()  # may need to change in future
            idx = int(self.env.get_depot_idx())
            or_route = file_or_route[self.batch_idx]
            or_cost = file_or_cost[self.batch_idx]
            action, base_line = self.predict(self.env.current_state, current_location, idx)
            sampled_value = self.env.G(action, current_location)
        else:
            self.env.reset()
            current_location = self.env.get_current_location()
            idx = int(self.env.get_depot_idx())
            action, base_line = self.predict(self.env.current_state, current_location, idx)
            sampled_value = self.env.G(action, current_location)
            if Config.REINFORCE == 0:
                or_model = OR_Tool(self.env.current_state, current_location, idx)
                or_route, or_cost = or_model.solve()
            else:
                or_route = np.zeros([Config.NUM_OF_CUSTOMERS+1], dtype=np.int32)
                or_cost = 1.0
        return action, base_line, sampled_value, or_route, or_cost, idx

    def run(self):
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            a_, b_, r_, ora_, orr_, idx_ = self.run_episode()
            x_ = self.env.current_state
            y_ = self.env.get_current_location()
            self.training_q.put(([x_], [y_], [a_], [ora_], [r_], [orr_], [idx_]))
