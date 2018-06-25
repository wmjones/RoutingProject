import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from OR_Tool import OR_Tool

state = np.vstack((np.random.rand(20, 2), np.array([0, 0])))
depot_location = state[-1]
depot_idx = len(state)-1
print(state)
or_model = OR_Tool(state, depot_location, depot_idx)
print(depot_location)
or_route, or_cost = or_model.solve()
or_route = np.asarray(or_route, dtype=np.int32)
or_cost = np.asarray(or_cost, dtype=np.float32)
print(or_route)
edges = np.concatenate((or_route[:-1].reshape(-1, 1), or_route[1:].reshape(-1, 1)), axis=1)
points = state
lc = LineCollection(points[edges])

fig = plt.figure()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.title('TSP 20')
fig.savefig('fig1.png')
plt.close(fig)


fig = plt.figure()
plt.gca().add_collection(lc)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.title('TSP 20 Solution')
fig.savefig('fig2.png')
plt.close(fig)
