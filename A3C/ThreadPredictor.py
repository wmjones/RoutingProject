from threading import Thread
import numpy as np
import time


from Config import Config


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.NUM_OF_CUSTOMERS+1, 2), dtype=np.float32)
        current_locations = np.zeros((Config.PREDICTION_BATCH_SIZE, 2), dtype=np.float32)
        depot_idx = np.zeros([Config.PREDICTION_BATCH_SIZE], dtype=np.int32)

        while not self.exit_flag:
            ids[0], states[0], current_locations[0], depot_idx[0] = self.server.prediction_q.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], states[size], current_locations[size], depot_idx[size] = self.server.prediction_q.get()
                size += 1

            batch = np.asarray(states[:size], dtype=np.float32)
            batch_locations = np.asarray(current_locations[:size], dtype=np.float32)
            batch_idx = np.asarray(depot_idx[:size], dtype=np.int32)
            a, b = self.server.model.predict(batch, batch_locations, batch_idx)

            for i in range(size):
                self.server.agents[ids[i]].wait_q.put((a[i], b[i]))
