import numpy as np
from Config import Config


class Environment:
    def __init__(self):
        # self.customer_locations = np.vstack((np.array([0, 0]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)*10))
        # self.distance_matrix = [[np.linalg.norm(self.customer_locations[i]-self.customer_locations[j])
        #                          for i in range(Config.NUM_OF_CUSTOMERS)]
        #                         for j in range(Config.NUM_OF_CUSTOMERS)]
        self.distance_matrix = []
        self.current_state = []
        self.depot_idx = 1

    def Distance(self, from_node, to_node):
        return self.distance_matrix[from_node][to_node]

    def reset(self):
        # self.current_state = np.vstack((np.array([.5, .5]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)))
        # self.current_state = np.vstack((np.array([0, 0]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)))
        self.current_state = np.vstack((np.random.rand(Config.NUM_OF_CUSTOMERS+1, 2)))
        # self.current_state = np.vstack((np.array([0, 0]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)*10))
        np.random.shuffle(self.current_state)
        # self.depot_idx = np.where(self.current_state[:, 0] == .5)[0][0]
        self.depot_idx = 0
        # self.depot_idx = np.where(self.current_state[:, 0] == 0)[0][0]
        self.distance_matrix = [[np.linalg.norm(self.current_state[i]-self.current_state[j])
                                 for i in range(Config.NUM_OF_CUSTOMERS+1)]
                                for j in range(Config.NUM_OF_CUSTOMERS+1)]

    def G(self, route, current_location):
        dist = np.linalg.norm(current_location - self.current_state[route[0]])  # current_location to first customer
        for i in range(0, len(route)-1):
            dist += self.Distance(route[i], route[i+1])
        # dist += self.Distance(route[len(route)], 0)  # return to depot
        # dist = dist + 6*(len(route) - len(np.unique(route)))  # can probably take out since it shouldnt happen anymore
        if len(route) - len(np.unique(route)) > 0:
            print("Error same location chosen twice")
        return(dist)

    def get_current_location(self):
        return self.current_state[self.depot_idx]

    def get_depot_idx(self):
        return(self.depot_idx)

    # def step(self, action):
    #     customer_location = self.customers[action]
    #     reward = -LA.norm(self.truck_location - customer_location)
    #     # print("customer_location={}".format(customer_location))
    #     self.truck_location = customer_location
    #     # self.remove_customer(action)
    #     self.total_reward += reward
    #     return reward

    # def nearest_customers(self, k):
    #     if(k > len(self.customers)):
    #         k = len(self.customers)
    #     dist = [LA.norm(x[0]-x[1]) for x in zip(self.customers, np.tile(self.truck_location, (len(self.customers), 1)))]
    #     sorted_idx = sorted(range(len(dist)), key=lambda k: dist[k])
    #     return np.array(self.customers)[sorted_idx[:k]].tolist()


# env = Environment()
# env.reset()
# route = np.arange(20)
# np.random.shuffle(route)
# print(env.G(route, env.current_state[0]))
