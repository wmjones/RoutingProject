import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import itertools

N = 8
data = np.random.rand(N, 2)*10
data = np.vstack((np.array([0, 0]), data))


class gen_matrix(object):
    def __init__(self, size):
        self.matrix = [[np.linalg.norm(data[i]-data[j]) for i in range(size)]
                       for j in range(size)]

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]


def objective(route):
    out = 0
    for i in range(len(route)-1):
        out += np.linalg.norm(data[route[i]]-data[route[i+1]])
    return(out)


def find_min_route(N):
    current_best = 1e5
    best_route = [range(N+2)]
    avg = 0
    count = 0
    for i in itertools.permutations(range(1, N), N-1):
        route = np.zeros(N+1, dtype=np.int32)
        for j in range(len(i)):
            route[j+1] = i[j]
        current = objective(route)
        avg += current
        count += 1
        if current_best >= current:
            current_best = current
            best_route = route
    return(best_route, current_best, avg/count)


routing = pywrapcp.RoutingModel(N, 1, 0)
search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
# search_parameters.first_solution_strategy = (
#     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.time_limit_ms = 30
matrix = gen_matrix(N)
dist_callback = matrix.Distance
routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
assignment = routing.SolveWithParameters(search_parameters)
# assignment = routing.Solve()

route = []
if assignment:
    print(assignment.ObjectiveValue())
    index = routing.Start(0)
    route.append(routing.IndexToNode(index))
    while not routing.IsEnd(index):
        index = assignment.Value(routing.NextVar(index))
        route.append(routing.IndexToNode(index))
print("Solution:")
print((route, objective(route)))

print("Truth:")
optimal_route, optimal_obj, avg_sol = find_min_route(N)
print((optimal_route, optimal_obj))

print("Average Solution:")
print(avg_sol)
