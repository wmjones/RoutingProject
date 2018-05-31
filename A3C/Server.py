import numpy as np
import time
from Config import Config
from Environment import Environment
from NetworkVP import NetworkVP
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


class Server:
    def __init__(self):
        self.model = NetworkVP(Config.DEVICE)
        self.training_step = 0

    def plot(self, step):
        env = Environment()
        batch_state, batch_or_cost, batch_or_route, batch_depot_location = env.next_batch()
        action, _ = self.model.predict([batch_state[0]], [batch_depot_location[0]])
        if Config.DIRECTION == 6:
            action = action[0]
        points = batch_state[0]
        edges = np.array([[19, action[0][0]]], dtype=np.int32)
        edges = np.append(edges, np.concatenate((action[0][:-1].reshape(-1, 1), action[0][1:].reshape(-1, 1)), axis=1), axis=0)
        # edges = np.append(edges, np.array([[action[0][-1], 19]], dtype=np.int32), axis=0) # taken out so i can see direction
        lc = LineCollection(points[edges])
        fig = plt.figure()
        plt.gca().add_collection(lc)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(points[:, 0], points[:, 1], 'ro')
        plt.title('Total Steps=' + str(step))
        fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) + '_MODEL_NAME_' + str(Config.MODEL_NAME) +
                    '_STEP_' + str(step) + '.png')
        plt.close(fig)
        if step > 0:
            or_route = batch_or_route[0]
            edges = np.array([[19, or_route[0]]], dtype=np.int32)
            edges = np.append(edges, np.concatenate((or_route[:-2].reshape(-1, 1), or_route[1:-1].reshape(-1, 1)), axis=1), axis=0)
            lc = LineCollection(points[edges])
            fig = plt.figure()
            plt.gca().add_collection(lc)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.plot(points[:, 0], points[:, 1], 'ro')
            fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) + '_MODEL_NAME_' + str(Config.MODEL_NAME) +
                        '_OPTIMAL' + '.png')
            plt.close(fig)

    def main(self):
        self.plot(self.model.get_global_step())
        self.env = Environment()
        t_end = time.time() + Config.RUN_TIME
        while time.time() < t_end:
            step = self.model.get_global_step()
            batch_state, batch_or_cost, batch_or_route, batch_depot_location = self.env.next_batch()
            if Config.REINFORCE == 0:
                self.model.train(state=batch_state, depot_location=batch_depot_location, or_action=batch_or_route)
            else:
                batch_pred_route, batch_pred_cost = self.model.predict(batch_state, batch_depot_location)
                batch_sampled_cost = self.env.cost(batch_state, batch_pred_route)
                self.model.train(state=batch_state, depot_location=batch_depot_location,
                                 sampled_cost=batch_sampled_cost, or_cost=batch_or_cost)
            if step % 1000 == 0:
                # self.plot(self.model.get_global_step())
                if Config.REINFORCE == 0:
                    batch_pred_route, batch_pred_cost = self.model.predict(batch_state, batch_depot_location)
                    batch_sampled_cost = self.env.cost(batch_state, batch_pred_route)
                self.model.summary(batch_state, batch_or_cost, batch_or_route, batch_depot_location,
                                   batch_pred_route, batch_pred_cost, batch_sampled_cost)
            if step % 100000 == 0 and step > 0:
                print("Saving Model...")
                self.model._model_save()
            # for i in range(len(batch_pred_route)):
            #     if len(batch_pred_route[i]) > len(np.unique(batch_pred_route[i])):
            #         self.model._model_save()
            #         print(step)
            #         sys.exit("Error same location chosen twice")

        self.plot(self.model.get_global_step())
        self.model.finish()
        print("total steps:", self.model.get_global_step())
