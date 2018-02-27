from multiprocessing import Queue
import numpy as np
import time

from Config import Config
# from Environment import Environment
from ProcessAgent import ProcessAgent
from NetworkVP import NetworkVP
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer


class Server:
    def __init__(self):
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
        # self.agents[-1].terminate()
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

    def main(self):
        self.trainer_count = Config.TRAINERS
        self.predictor_count = Config.PREDICTORS
        self.agent_count = Config.AGENTS

        for _ in np.arange(0, Config.TRAINERS):
            self.add_trainer()
        for _ in np.arange(0, Config.PREDICTORS):
            self.add_predictor()
        for _ in np.arange(0, Config.AGENTS):
            self.add_agent()

        # if Config.PLAY_MODE:
        #     for trainer in self.trainers:
        #         trainer.endabled = False

        # while self.stats.episode_count.value < Config.EPISODES:
        #     # step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)  # not sure if i need this
        #     self.model.learning_rate = Config.LEARNING_RATE
        #     # self.model.beta = Config.BETA_START + beta_multiplier * step

        time.sleep(Config.RUN_TIME)

        if Config.TRAIN:
            self.model.finish()

        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
        print("total steps:", self.model.get_global_step())
