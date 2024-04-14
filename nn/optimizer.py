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

    def is_improved(prediction, )
    
    def train_network(self, x_train=[], y_train=[]):
        if len(x_train) != len(y_train):
            print("Shape error")
            return
        else:
            for index, each_mini_batch in enumerate(x_train):
                predicted = self.network.forward_pass(each_mini_batch)


