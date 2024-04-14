from typing import List
import numpy as np
import global_config as gb


class Neuron:
    def __init__(self, no_of_weights) -> None:
        self.weights = []
        self.bias = 0
        self.buffer = 0
        for _ in range(no_of_weights):
            self.weights.append(np.random.uniform(gb.config["random_value_min"], gb.config["random_value_max"]))
    
    def forward_pass(self, input_data=[]):
        if len(input_data) != len(self.weights):
            print("Shape error")
            return
        else:
            for _ in range(len(input_data)):
                self.buffer += self.weights[_] * input_data[_]
            self.buffer += self.bias
            return max(0, self.buffer)
    



class Dense_Layer:
    def __init__(self, no_of_neurons=0) -> None:
        self.no_of_neurons = no_of_neurons
        self.neurons: List[Neuron] = []
        self.buffer = []
    def build(self, no_of_weight_in_each_neuron=0):
        for _ in range(self.no_of_neurons):
            self.neurons.append(Neuron(no_of_weights=no_of_weight_in_each_neuron))

    def forward_pass(self, input_data=[]):
        self.buffer.clear()
        for _ in range(self.no_of_neurons):
            self.buffer.append(self.neurons[_].forward_pass(input_data=input_data))
        return self.buffer


class Sequential_Neural_Network:
    def __init__(self) -> None:
        self.layers: List[Dense_Layer] = []
        self.buffer = []

    def add(self, layer=Dense_Layer()):
        self.layers.append(layer)
    def build(self, input_length=0):
        self.layers[0].build(input_length)

        for _ in range(1, len(self.layers)):
            self.layers[_].build(self.layers[_ - 1].no_of_neurons)

    def forward_pass(self, input_data=[]):
        self.buffer.clear()
        self.buffer = input_data
        for _ in range(len(self.layers)):
            self.buffer = self.layers[_].forward_pass(self.buffer)
        return self.buffer



network = Sequential_Neural_Network()
network.add(Dense_Layer(2))
network.add(Dense_Layer(3))
network.add(Dense_Layer(3))
network.build(input_length=2)
print("Success")
print(network.forward_pass([3, 5]))
