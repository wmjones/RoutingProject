import math
import random
import numpy as np
import matplotlib.pyplot as plt

T = 480
speed = 300.0
dronespeed = 2*speed
veh = 5
drones = 10
deadline = 240
threshold = 10
served_num = 0
total_trials = 200
total_requests = 0

theta11 = 30
theta12 = -2
theta21 = 1
theta22 = -1
parameter = [theta11, theta12, theta21, theta22]
bias1 = 1
bias2 = 1


# define Order class
class Order(object):
    def __init__(self, number, time, xcoord, ycoord, position, arrival):
        self.number = number
        self.time = time
        self.x = xcoord
        self.y = ycoord
        self.position = position
        self.arrival = arrival


# for each new order: create potential Routes_Cand
def dist(order1, order2, time, speed):
    x1 = order1.x
    x2 = order2.x
    y1 = order1.y
    y2 = order2.y
    a = x1 - x2
    b = y1 - y2
    m = math.sqrt(a*a + b*b)
    d = (1.0*m)/speed
    c = int(max(math.ceil(d), 1) + 2)
    return c


def duration(tour, time, speed):
    tour_duration = 0;
    for i in range(0, len(tour)-1):
        tour_duration = tour_duration + int(dist(tour[i], tour[i+1], time, speed))
    return tour_duration


def Routing_drone(arrival, depot, order, time, dronespeed, deadline):
    arrival_temp = []
    for i in range(0, len(arrival)):
        arrival_temp.append(0)
    dr = 0
    for j in range(0, len(arrival)):
        arrival_temp[j] = arrival[j]
    for j in range(0, len(arrival_temp)):
        if arrival_temp[j] <= time and dr == 0:
            dr = 1
            arrival_temp[j] = time + 2*dist(depot, order, time, dronespeed) + 30
            j = len(arrival_temp)
    if dr == 0:
        for j in range(0, len(arrival_temp)):
            if arrival_temp[j] <= time + deadline - dist(depot, order, time, dronespeed) + 30 and dr == 0:
                dr = 1
                arrival_temp[j] = arrival_temp[j] + 2*dist(depot, order, time, dronespeed) + 30
                j = len(arrival_temp)
    if dr == 0:
        arrival_temp[0] = -1
    return arrival_temp


def Copy_Order(order):
    order_copy = Order(order.number, order.time, order.x, order.y, order.position, order.arrival)
    return order_copy


# each car's route
def Copy_Route(Route):
    route_copy = []
    for i in range(0, len(Route)):
        route_copy.append(Copy_Order(Route[i]))
    return route_copy


def Copy_Routing(Routing_org):
    Routing = []
    for i in range(0, len(Routing_org)):
        Routing.append(Copy_Route(Routing_org[i]))
    return Routing


def Update_Routes(Route_org, time):
    for i in range(0, len(Route_org)):
        if Route_org[i][0].arrival <= time and len(Route_org[i]) > 2:
            Route_org[i].pop(0)
    return Route_org


def Feasible(Route, deadline):
    for i in range(0, len(Route)):
        if Route[i].arrival - Route[i].time - deadline > 0 and Route[i].number > 0:
            return False
    return True


def Route(Route_org, order_org, time, speed):
    Route_temp = Copy_Route(Route_org)
    position_depot = -1
    dist_max = 10000
    position_order = -1
    for j in range(0, len(Route_temp)-1):
        if Route_temp[j].number == 0:
            position_depot = j
    if position_depot == -1:
        Route_temp.insert(len(Route_temp), order_org)
        Route_temp.insert(len(Route_temp), depot)
    else:
        dist_max = 10000
        for l in range(position_depot, len(Route_temp)-1):
            dist_temp = dist(order_org, Route_org[l], time, speed) +\
                        dist(order_org, Route_org[l+1], time, speed) - \
                        dist(Route_org[l], Route_org[l+1], time, speed)
            if dist_temp < dist_max:
                dist_max = dist_temp
                order_org.position = l+1
                position_order = l+1
        Route_temp.insert(position_order, order_org)
    arrival_temp = Route_temp[0].arrival
    for i in range(0, len(Route_temp)):
        Route_temp[i].arrival = arrival_temp
        if i < len(Route_temp)-1:
            arrival_temp = arrival_temp + dist(Route_temp[i], Route_temp[i+1], time, speed)
    return Route_temp


def Routing(Route_org, order_org, time, speed, deadline):
    Route_empty = []
    dist_max = 10000
    tour_order = -1
    Route_best = []
    for i in range(0, len(Route_org)):
        Route_temp = Copy_Route(Route_org[i])
        Route_temp = Route(Route_temp, order_org, time, speed)
        if Feasible(Route_temp, deadline):
            if duration(Route_temp, time, speed) - duration(Route_org[i], time, speed) < dist_max:
                dist_max = duration(Route_temp, time, speed) - duration(Route_org[i], time, speed)
                tour_order = i
                Route_best = Copy_Route(Route_temp)
    if tour_order == -1:
        return Route_empty
    else:
        Route_org[tour_order] = Route_best
    return Route_org


# This is for 'No Neural Network'
def Select(veh_feasible, drone_feasible, dist, threshold):
    if veh_feasible > 0 and drone_feasible < 1:
        return int(1)
    if drone_feasible > 0 and veh_feasible < 1:
        return int(2)
    if veh_feasible > 0 and dist < threshold:
        return int(1)
    if drone_feasible > 0 and dist >= threshold:
        return int(2)
    return int(0)


def Sigmoid(x):
    y = 1/(1+math.exp(-x))
    return y


def NN_function(parameter, norm_distance):
    input_h1 = parameter[1]*norm_distance
    output_hl = Sigmoid(input_h1)
    # print('the output of the hidden layer is %f.' % hl)
    input_ol = output_hl
    output_ol = input_ol
    # print('the outout of the NN is %f.' % ol)
    if output_ol >= 0.5:
        return int(2)
    if output_ol < 0.5:
        return int(1)


# This is for 'with NN'
def Select2(veh_feasible, drone_feasible, normd, parameter):
    if veh_feasible > 0 and drone_feasible < 1:
        return int(1)
    if drone_feasible > 0 and veh_feasible < 1:
        return int(2)
    if veh_feasible > 0 and drone_feasible > 0:
        if NN_function(parameter, normd) == 1:
            return int(1)
        else:
            return int(2)
    else:
        return int(0)


def Copy_parameter(parameter):
    copy = []
    for i in range(0, len(parameter)):
        copy.append(parameter[i])
    return copy


def Update_parameter(lr, parameter, gradient):
    for i in range(0, len(parameter)):
        parameter[i] = parameter[i] - lr*gradient[i]
    return parameter


# define depot
depot = Order(0, 0, 5000, 5000, 0, 0)
orderlist = []

for trial in range(1, total_trials + 1):
    temp_served_num = 0
    order_list = []
    file_num = str(trial)
    f = open('/Users/huohuodad/Desktop/pythonfiles/drones/480_G_500_500_%s.txt' % file_num, 'r')
    order_list = [line.split('_') for line in f.read().split('\n')]
    f.close
    orderlist = []
    total_requests = total_requests + len(order_list)-1
    for i in range(0, len(order_list)-1):
        number = int(order_list[i][0])
        time = int(order_list[i][1])
        xcoord = float(order_list[i][2])
        ycoord = float(order_list[i][3])
        orderlist.append(Order(number, time, xcoord, ycoord, 0, 0))
    arrival = []
    for i in range(0, drones):
        arrival.append(0)
    Routes = []
    for i in range(0, veh):
        route = []
        route.append(depot)
        route.append(depot)
        Routes.append(route)
    for time in range(0, T):
        NewOrders = []
        drone_feasible = 0
        veh_feasible = 0
        veh_num, drone_num = 0, 0
        for i in range(0, len(orderlist)):
            if orderlist[i].time == time:
                NewOrders.append(orderlist[i])
        for i in range(0, len(NewOrders)):
            Routes_Cand = []
            Routes_Cand = Copy_Routing(Routes)
            Routes_Cand = Routing(Routes_Cand, NewOrders[i], time, speed, deadline)
            if len(Routes_Cand) > 0:
                veh_feasible = 1
            arrival_cand = []
            for k in range(0, drones):
                arrival_cand.append(0)
            arrival_cand = Routing_drone(arrival, depot, NewOrders[i], time, dronespeed, deadline)
            if arrival_cand[0] > 0:
                drone_feasible = 1
            distance = dist(NewOrders[i], depot, time, speed)
            decision = Select(veh_feasible, drone_feasible, distance, threshold)
            # Here uses 'Select2' if we want to implement NN
            if decision > 0:
                temp_served_num = temp_served_num + 1
            if decision == 1:
                veh_num = veh_num + 1  # veh_num isnt previously defined
                Routes = Copy_Routing(Routes_Cand)
            elif decision == 2:
                drone_num = drone_num + 1  # drone_num isnt previously defined
                arrival = Routing_drone(arrival, depot, NewOrders[i], time, dronespeed, deadline)
        Routes = Update_Routes(Routes, time)
    served_num = served_num + temp_served_num
solution_quality = served_num / total_requests
print(solution_quality)
# print('The total number of orders served is %d.' % served_num)
# print('The number of orders served by veh\'s is %d.' % veh_num)
# print('The number of orders served by drones is %d.' % drone_num)

x = np.random.uniform(0,1,10)
y = np.zeros((len(x),))

for i in range(len(x)):
    if(x[i] > .5):
        y[i] = 1
    else:
        y[i] = 0
