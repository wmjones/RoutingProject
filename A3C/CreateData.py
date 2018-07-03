import numpy as np
from OR_Tool import OR_Tool
from sklearn.decomposition import PCA

file_state = open('data_state_20_00_pca.npy', 'wb')
file_or_route = open('data_or_route_20_00_pca.npy', 'wb')
file_or_cost = open('data_or_cost_20_00_pca.npy', 'wb')
for i in range(10):
    print(i)
    # depot_location = np.array([.5, .5])
    # data_state = np.vstack((np.random.rand(19, 2), np.array([.5, .5])))
    # depot_idx = int(np.where(data_state[:, 0] == .5)[0][0])
    depot_location = np.array([0, 0])
    data_state = np.random.rand(20, 2)
    pca = PCA(2)
    data_state = pca.fit_transform(data_state)
    data_state = np.vstack((np.array([0, 0]), data_state))
    # depot_idx = int(np.where(data_state[:, 0] == 0)[0][0])
    depot_idx = 0
    or_model = OR_Tool(data_state, depot_location, depot_idx)
    or_route, or_cost = or_model.solve()
    or_route = np.asarray(or_route, dtype=np.int32)
    or_cost = np.asarray(or_cost, dtype=np.float32)
    data_state = np.asarray([data_state], dtype=np.float32)
    if i == 0:
        out_data_state = data_state
        out_data_or_route = or_route
        out_data_or_cost = or_cost
    else:
        out_data_state = np.append(out_data_state, data_state, axis=0)
        out_data_or_route = np.vstack((out_data_or_route, or_route))
        out_data_or_cost = np.append(out_data_or_cost, or_cost)
np.save(file_state, out_data_state)
np.save(file_or_route, out_data_or_route)
np.save(file_or_cost, out_data_or_cost)
file_state.close()
file_or_route.close()
file_or_cost.close()
