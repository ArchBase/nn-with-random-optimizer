import global_config as gb
from typing import List
import numpy as np

class Mean_Squared_Error:
    def __init__(self) -> None:
        pass
    def get_loss_array(self, predicted_values=[], ground_truth_values=[]):
        if len(predicted_values) != len(ground_truth_values):
            print("Shape error")
        else:
            buffer = []
            for _ in range(len(ground_truth_values)):
                buffer.append(np.sqrt(ground_truth_values[_] - predicted_values[_]))
            return buffer


