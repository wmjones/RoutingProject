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
            while batch_size < Config.TRAINING_MIN_BATCH_SIZE:
                x_, y_, a_, ora_, r_, orr_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_
                    y__ = y_
                    a__ = a_
                    r__ = r_
                    ora__ = ora_
                    orr__ = orr_
                else:
                    x__ = np.vstack((x__, x_))
                    y__ = np.vstack((y__, y_))
                    a__ = np.vstack((a__, a_))
                    r__ = np.vstack((r__, r_))
                    ora__ = np.vstack((ora__, ora_))
                    orr__ = np.vstack((orr__, orr_))
                batch_size += 1
            # print(ora__, orr__)
            self.server.train_model(x__, y__, a__, ora__, r__, orr__, self.id)

            #     print("state: ", x__)
            #     print("value: ", r__)
            #     self.exit_flag = True
            # print("hello from thread ", self.id)
            # print("exit", self.exit_flag)
            # print(x__)
