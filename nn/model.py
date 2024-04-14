import neural_network
import optimizer
import loss_functions
import global_config as gb
from typing import List
import numpy as np
import random




class model:
    def __init__(self) -> None:
        pass

network = neural_network.Sequential_Neural_Network()
network.add(neural_network.Dense_Layer(2))
network.add(neural_network.Dense_Layer(3))
network.add(neural_network.Dense_Layer(1))
network.build(input_length=2)

lsfn = loss_functions.Mean_Squared_Error()

optm = optimizer.Flash_Optimizer(network, lsfn)

x_train = [[1, 2], [5, 2], [7, 9], [0, 3], [4, 2]]
y_train = [[3], [6], [7], [4], [10]]

u = network.get_parameter_array()
print("Hai")
print(u)
for _ in range(len(u)):
    u[_] = u[_] + np.random.uniform(gb.config["optimizer_random_min"], gb.config["optimizer_random_max"])
            
    network.apply_parameter(u)


print("YOYO GUY")
print(network.get_parameter_array())
#optm.train_network(x_train, y_train)


