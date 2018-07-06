import numpy as np
import time
from Config import Config
from Environment import Environment
from NetworkVP import NetworkVP
# import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
np.set_printoptions(threshold=np.nan)


class Server:
    def __init__(self):
        self.training_step = 0

    def plot(self, state, action, step):
        points = state
        edges = np.array([[Config.NUM_OF_CUSTOMERS, action[0]]], dtype=np.int32)
        edges = np.append(edges, np.concatenate((action[:-1].reshape(-1, 1), action[1:].reshape(-1, 1)), axis=1), axis=0)
        lc = LineCollection(points[edges])
        fig = plt.figure()
        plt.gca().add_collection(lc)
        if Config.USE_PCA == 1:
            lower_bound = -1
        else:
            lower_bound = 0
        plt.xlim(lower_bound, 1)
        plt.ylim(lower_bound, 1)
        plt.plot(points[:, 0], points[:, 1], 'ro')
        plt.title('Total Steps=' + str(step-1))
        fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) + '_MODEL_NAME_' + str(Config.MODEL_NAME) +
                    '_STEP_' + str(step-1) + '.png')
        plt.close(fig)
        # if True:
        #     edges = np.array([[19, or_route[0]]], dtype=np.int32)
        #     edges = np.append(edges, np.concatenate((or_route[:-2].reshape(-1, 1), or_route[1:-1].reshape(-1, 1)), axis=1), axis=0)
        #     edges = np.concatenate((np.arange(0, 20).reshape(-1, 1), np.arange(1, 21).reshape(-1, 1)), axis=1)
        #     print(edges)
        #     lc = LineCollection(points[edges])
        #     fig = plt.figure()
        #     plt.gca().add_collection(lc)
        #     plt.xlim(-1, 1)
        #     plt.ylim(-1, 1)
        #     plt.plot(points[:, 0], points[:, 1], 'ro')
        #     fig.savefig(str(Config.PATH) + 'figs/TSP_' + str((Config.NUM_OF_CUSTOMERS+1)) +
        #                 '_MODEL_NAME_' + str(Config.MODEL_NAME) +
        #                 '_OPTIMAL' + '.png')
        #     plt.close(fig)

    def main(self):
        if Config.USE_PPO == 1:
            Config.SEQUENCE_COST = 1
        self.env = Environment()
        self.model = NetworkVP(Config.DEVICE, DECODER_TYPE=0)
        batch_state, batch_or_cost, batch_or_route, batch_depot_location = self.env.next_batch(Config.TRAINING_MIN_BATCH_SIZE)
        test_state = np.asarray([batch_state[0]], dtype=np.float32)
        test_depot_location = batch_depot_location[0]
        t_end = time.time() + Config.RUN_TIME
        step = -1
        while time.time() < t_end:
            step += 1
            batch_state, batch_or_cost, batch_or_route, batch_depot_location = self.env.next_batch(Config.TRAINING_MIN_BATCH_SIZE)
            old_probs = np.zeros((batch_state.shape[0], batch_state.shape[1]-1, batch_state.shape[1]-1))
            if Config.REINFORCE == 0:
                self.model.train(state=batch_state, depot_location=batch_depot_location, or_action=batch_or_route)
            else:
                if Config.USE_PPO == 1:
                    old_probs = self.model.PPO(state=batch_state, depot_location=batch_depot_location)
                    old_probs = np.clip(old_probs, 1e-6, 1)
                batch_pred_route, batch_pred_cost = self.model.predict(batch_state, batch_depot_location)
                batch_sampled_cost = self.env.cost(batch_state, batch_pred_route)
                if step == 0:
                    self.model.init_MA(batch_sampled_cost)
                if Config.USE_PPO == 0:
                    self.model.train(state=batch_state, depot_location=batch_depot_location,
                                     sampled_cost=batch_sampled_cost, or_cost=batch_or_cost)
                else:
                    for i in range(Config.NUM_PPO_EPOCH):
                        self.model.train(state=batch_state, depot_location=batch_depot_location,
                                         sampled_cost=batch_sampled_cost, or_cost=batch_or_cost, old_probs=old_probs)

            if step % 1000 == 0:
                test_pred_route, _ = self.model.predict(test_state, [test_depot_location])
                self.plot(test_state[0], test_pred_route[0][0], self.model.get_global_step())
                print("Saving Model...")
                self.model._model_save()
                print("Done Saving Model")
                batch_state, batch_or_cost, batch_or_route, batch_depot_location = self.env.next_batch(10)
                batch_pred_route, batch_pred_cost = self.model.predict(batch_state, batch_depot_location)
                if Config.DIRECTION == 6:
                    self.eval_model = NetworkVP(Config.DEVICE, DECODER_TYPE=1)
                    batch_eval_pred_route, batch_eval_pred_cost = self.eval_model.predict(batch_state, batch_depot_location)
                elif Config.SAMPLING == 1:
                    self.eval_model = NetworkVP(Config.DEVICE, DECODER_TYPE=2)
                    batch_eval_pred_route, batch_eval_pred_cost = self.eval_model.predict(batch_state, batch_depot_location, 2)
                else:
                    batch_eval_pred_route = batch_pred_route
                batch_eval_sampled_cost = self.env.cost(batch_state, batch_eval_pred_route)
                batch_sampled_cost = self.env.cost(batch_state, batch_pred_route)
                if Config.SAMPLING == 1 or Config.DIRECTION == 6:
                    batch_sampled_cost = batch_eval_sampled_cost
                self.model.summary(batch_state, batch_or_cost, batch_or_route, batch_depot_location,
                                   batch_pred_cost, batch_sampled_cost, old_probs)
                if Config.SEQUENCE_COST == 1:
                    batch_eval_sampled_cost = batch_eval_sampled_cost[:, 0]
                    batch_sampled_cost = batch_sampled_cost[:, 0]
                # self.plot(batch_state[0], batch_pred_route[0][0], self.model.get_global_step())

                print("batch_or_route:")
                print(batch_or_route)
                print("batch_pred_route:")
                print(batch_pred_route)
                print("batch_eval_pred_route:")
                print(batch_eval_pred_route)
                print("avg_batch_or_cost:")
                print(np.mean(batch_or_cost))
                print("avg_batch_pred_cost:")
                print(np.mean(batch_sampled_cost))
                print("avg_batch_eval_sampled_cost:")
                print(np.mean(batch_eval_sampled_cost))

            # for i in range(len(batch_pred_route)):
            #     if len(batch_pred_route[i]) > len(np.unique(batch_pred_route[i])):
            #         self.model._model_save()
            #         print(step)
            #         sys.exit("Error same location chosen twice")

        self.model.finish()
        print("total steps:", self.model.get_global_step())
