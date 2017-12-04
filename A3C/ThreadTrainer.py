from threading import Thread
import numpy as np

from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)
        self.id = id
        self.server = server
        self.exit_flag = False
        self.count = 0

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                x_, a_, r_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_
                    a__ = a_
                    r__ = r_
                else:
                    x__ = np.vstack((x__, x_))
                    a__ = np.vstack((a__, a_))
                    r__ = np.vstack((r__, r_))
                batch_size += 1
            self.server.train_model(x__, a__, r__, self.id)
            #     print("state: ", x__)
            #     print("value: ", r__)
            #     self.exit_flag = True
            # print("hello from thread ", self.id)
            # print("exit", self.exit_flag)
            # print(x__)
