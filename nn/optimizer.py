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
        self.prev_cost = None

        self.new_params = None
        self.new_predicted = None
        self.new_cost = None

    def train_on_batch(self, x_train, y_train=[]):
        print("TTTTTTTTTTTT")
        print(self.network.get_parameter_array())
        self.prev_params = self.network.get_parameter_array()
        
        self.new_params = self.prev_params
        
        self.prev_predicted = self.network.forward_pass(x_train)
        print("\n\n\nBEFORE\n\n\n")
        print(self.prev_params)
        print("\n\n\nAFTER\n\n\n")
        self.prev_cost = self.loss_function.get_cost(predicted_values=self.prev_predicted, ground_truth_values=y_train)
        _ = 0
        
        print(f"\n\nLength of \n\n")
        while _ < gb.config["max_random_patience"]:
            print("Running" + str(_))
            for _ in range(len(self.prev_params)):
                self.new_params[_] = self.prev_params[_] + np.random.uniform(gb.config["optimizer_random_min"], gb.config["optimizer_random_max"])
            
            self.network.apply_parameter(self.new_params)

            self.new_predicted = self.network.forward_pass(x_train)
            self.new_cost = self.loss_function.get_cost(predicted_values=self.new_predicted, ground_truth_values=y_train)

            if self.new_cost <= self.prev_cost:
                self.prev_params = self.new_params
                self.prev_predicted = self.new_predicted
                self.prev_cost = self.new_cost
            else:
                self.network.apply_parameter(self.prev_params)
            _ += 1







    def train_network(self, x_train=[], y_train=[]):
        if len(x_train) != len(y_train):
            print("Shape error")
            return
        else:
            for index, x in enumerate(x_train):
                self.train_on_batch(x, y_train[index])



