from multiprocessing import Queue
import numpy as np
import time

from Config import Config
from Environment import Environment
from ProcessAgent import ProcessAgent
from NetworkVP import NetworkVP
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from OR_Tool import OR_Tool
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


class Server:
    def __init__(self):
        self.training_q = Queue(maxsize=Config.TRAINING_MIN_BATCH_SIZE)
        self.training_q = Queue(maxsize=50)
        self.prediction_q = Queue(maxsize=20)

        self.model = NetworkVP(Config.DEVICE)
        self.training_step = 0

        self.agents = []
        self.predictors = []
        self.trainers = []

    def add_agent(self):
        self.agents.append(ProcessAgent(len(self.agents), self.prediction_q, self.training_q))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = 1
        self.agents[-1].join()
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join(0.1)
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join(0.1)
        self.trainers.pop()

    def train_model(self, x__, y__, a__, ora__, r__, orr__, idx__, trainer_id):
        self.model.train(x__, y__, a__, ora__, r__, orr__, idx__, trainer_id)
        self.training_step += 1

    def or_model(self, x__, trainer_id):
        return(self.or_model.solve(x__))

    def plot(self, step):
        env = Environment()
        env.reset()
        action, _ = self.model.predict([env.current_state], [env.get_current_location()], [env.depot_idx])
        points = env.current_state
        edges = np.array([[19, action[0][0]]], dtype=np.int32)
        edges = np.append(edges, np.concatenate((action[0][:-1].reshape(-1, 1), action[0][1:].reshape(-1, 1)), axis=1), axis=0)
        edges = np.append(edges, np.array([[action[0][-1], 19]], dtype=np.int32), axis=0)
        lc = LineCollection(points[edges])
        fig = plt.figure()
        plt.gca().add_collection(lc)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(points[:, 0], points[:, 1], 'ro')
        plt.title('Total Steps=' + str(step))
        fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) + '_' + 'MODEL_NAME_' + str(Config.MODEL_NAME) +
                    '_STEP_' + str(step) + '.png')
        if True:
            or_model = OR_Tool(env.current_state, env.get_current_location(), int(env.depot_idx))
            or_route, or_cost = or_model.solve()
            or_route = np.asarray(or_route)
            edges = np.array([[19, or_route[0]]], dtype=np.int32)
            edges = np.append(edges, np.concatenate((or_route[:-1].reshape(-1, 1), or_route[1:].reshape(-1, 1)), axis=1), axis=0)
            lc = LineCollection(points[edges])
            fig = plt.figure()
            plt.gca().add_collection(lc)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.plot(points[:, 0], points[:, 1], 'ro')
            fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) + '_Optimal' + '.png')

    def main(self):
        self.trainer_count = Config.TRAINERS
        self.predictor_count = Config.PREDICTORS
        self.agent_count = Config.AGENTS

        self.plot(self.model.get_global_step())

        for _ in np.arange(0, Config.TRAINERS):
            self.add_trainer()
        for _ in np.arange(0, Config.PREDICTORS):
            self.add_predictor()
        for _ in np.arange(0, Config.AGENTS):
            self.add_agent()

        time.sleep(Config.RUN_TIME)

        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()

        if Config.TRAIN:
            self.plot(self.model.get_global_step())
            self.model.finish()
        print("total steps:", self.model.get_global_step())
