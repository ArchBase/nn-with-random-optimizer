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
        self.prev_params = self.network.get_parameter_array()
        self.network.apply_parameter(self.prev_params)
        self.prev_predicted = self.network.forward_pass(x_train)
        self.prev_reward = self.loss_function.get_reward(self.prev_predicted, y_train)
        self.new_params = self.prev_params.copy()
        #print(f"trained on batch x: {x_train} y:{y_train} reward:{self.prev_predicted} params:{self.prev_params}")

        #self.new_params = self.randomly_modify_array(self.prev_params.copy())
        print(f"params : {self.new_params}")
        self.network.apply_parameter(self.new_params)
        self.new_predicted = self.network.forward_pass(x_train)
        self.new_reward = self.loss_function.get_reward(self.new_predicted, y_train)
        print(f"reward: {self.prev_reward - self.new_reward}")
        if self.prev_reward - self.new_reward == 0:
            print(f"REWARD NOT:{self.prev_predicted} and {self.new_predicted}")
        

        






    def train_network(self, x_train=[], y_train=[]):
        if len(x_train) != len(y_train):
            print("Shape error")
            return
        else:
            for index, x in enumerate(x_train):
                #print(f"Training on x:{x}, y:{y_train[index]}")
                self.train_on_batch(x, y_train[index])



