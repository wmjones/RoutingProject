import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class gen_matrix(object):
    def __init__(self, size, data):
        self.matrix = [[np.linalg.norm(data[i]-data[j]) for i in range(size)]
                       for j in range(size)]

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]


class OR_Tool:
    def __init__(self, data):
        self.data = data
        self.size = data.shape[0]

    def objective(self, route):
        out = 0
        for i in range(len(route)-1):
            out += np.linalg.norm(self.data[route[i]]-self.data[route[i+1]])
        return(out)

    def solve(self):
        routing = pywrapcp.RoutingModel(self.size, 1, 0)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit_ms = 30
        matrix = gen_matrix(self.size, self.data)
        dist_callback = matrix.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        assignment = routing.SolveWithParameters(search_parameters)
        route = []
        if assignment:
            index = routing.Start(0)
            route.append(routing.IndexToNode(index))
            while not routing.IsEnd(index):
                index = assignment.Value(routing.NextVar(index))
                route.append(routing.IndexToNode(index))
        return(self.objective(route))
