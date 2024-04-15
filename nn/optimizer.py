import global_config as gb
from typing import List
import numpy as np
import neural_network
import loss_functions
import random

class Flash_Optimizer:
    def __init__(self, network=neural_network.Sequential_Neural_Network(), loss_function=loss_functions.Mean_Squared_Error()) -> None:
        self.network = network
        self.loss_function = loss_function
        self.prev_params = None
        self.prev_predicted = None
        self.prev_reward = None

        self.new_params = None
        self.new_predicted = None
        self.new_reward = None
    
    def randomly_modify_array(self, array=[]):
        for _ in range(len(array)):
            array[_] += np.random.uniform(gb.config["optimizer_random_min"], gb.config["optimizer_random_max"])
        return array

    def train_on_batch(self, x_train=[], y_train=[]):
        print(x_train)
        print("old")
        self.prev_params = self.network.get_parameter_array()
        self.prev_predicted = self.network.forward_pass(x_train.copy())
        print(self.prev_predicted)

        print("new")
        self.new_params = self.randomly_modify_array(self.prev_params.copy())
        self.network.apply_parameter(self.new_params)
        self.new_predicted = self.network.forward_pass(x_train.copy())
        print(self.new_predicted)
        print("\n\n********************")






    def train_network(self, x_train=[], y_train=[]):
        if len(x_train) != len(y_train):
            print("Shape error")
            return
        else:
            for index, x in enumerate(x_train):
                #print(f"Training on x:{x}, y:{y_train[index]}")
                self.train_on_batch(x, y_train[index])



