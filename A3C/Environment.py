import numpy as np
from Config import Config


class Environment:
    def __init__(self):
        # self.customer_locations = np.vstack((np.array([0, 0]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)*10))
        # self.distance_matrix = [[np.linalg.norm(self.customer_locations[i]-self.customer_locations[j])
        #                          for i in range(Config.NUM_OF_CUSTOMERS)]
        #                         for j in range(Config.NUM_OF_CUSTOMERS)]
        self.customer_locations = []
        self.distance_matrix = []
        self.current_state = self.customer_locations

    def Distance(self, from_node, to_node):
        return self.distance_matrix[from_node][to_node]

    def reset(self):
        self.customer_locations = np.vstack((np.array([0, 0]), np.random.rand(Config.NUM_OF_CUSTOMERS, 2)*10))
        # self.customer_locations = np.random.rand(Config.NUM_OF_CUSTOMERS, 2)*10
        self.distance_matrix = [[np.linalg.norm(self.customer_locations[i]-self.customer_locations[j])
                                 for i in range(Config.NUM_OF_CUSTOMERS+1)]
                                for j in range(Config.NUM_OF_CUSTOMERS+1)]
        self.current_state = self.customer_locations

    def G(self, route, current_location):
        dist = np.linalg.norm(current_location - self.customer_locations[route[0]])  # current_location to first customer
        for i in range(0, len(route)-1):
            dist += self.Distance(route[i], route[i+1])
        # dist += self.Distance(route[len(route)], 0)  # return to depot
        return(dist)

    def get_current_location(self):
        return np.array([0.0, 0.0], dtype=np.float32)

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
