import numpy as np
from Config import Config
import sys


class Environment:
    def __init__(self):
        self.file_state = np.load('data_state_00.npy', 'r')
        self.file_or_route = np.load('data_or_route_00.npy', 'r')
        self.file_or_cost = np.load('data_or_cost_00.npy', 'r')
        self.batch_size = Config.TRAINING_MIN_BATCH_SIZE

    def next_batch(self):
        batch_idx = [np.random.randint(10000) for i in range(self.batch_size)]
        state_batch = []
        for i in range(self.batch_size):
            state_i = self.file_state[batch_idx[i], :]
            state_batch.append(state_i)
        state_batch = np.asarray(state_batch)
        route_batch = []
        for i in range(self.batch_size):
            route_i = self.file_or_route[batch_idx[i], :]
            route_batch.append(route_i)
        route_batch = np.asarray(route_batch, dtype=np.int32)
        cost_batch = []
        for i in range(self.batch_size):
            cost_i = self.file_or_cost[batch_idx[i]]
            cost_batch.append(cost_i)
        cost_batch = np.asarray(cost_batch).reshape(-1, 1)
        depot_location_batch = np.tile([19], self.batch_size)
        return(state_batch, cost_batch, route_batch, depot_location_batch)

    def cost(self, raw_state, action):
        costs = []
        for i in range(raw_state.shape[0]):
            best_cost = 1e10
            for j in range(Config.BEAM_WIDTH):
                state = np.take(raw_state[i], action[i][j], axis=0)
                cost = np.sum(np.sqrt(np.sum(np.square(state[1:]-state[:-1]), axis=1)))
                cost += np.sum(np.sqrt(np.sum(np.square(state[0], np.array([0, 0], dtype=np.float32)))))
                cost += np.sum(np.sqrt(np.sum(np.square(state[-1], np.array([0, 0], dtype=np.float32)))))
                if cost < best_cost:
                    best_cost = cost
            costs.append(cost)
        costs = np.asarray(costs)
        return(costs.reshape(-1, 1))
