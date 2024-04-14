from typing import List


class Neuron:
    def __init__(self, no_of_weights) -> None:
        self.weights = []
        self.bias = 0
        self.buffer = []
    
    def forward_pass(self, input_data=[]):
        self.buffer.clear()
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
        self.neurons: List[Dense_Layer] = []
        self.buffer = []
    def build(self, no_of_weight_in_each_neuron=0):
        for _ in range(self.no_of_neurons):
            self.neurons.append(Neuron(no_of_weights=no_of_weight_in_each_neuron))

    def forward_pass(self, input_data=[]):
        self.buffer.clear()
        for _ in range(self.no_of_neurons):
            self.buffer.append(self.neurons[_].forward_pass(input_data=input_data))


class Sequential_Neural_Network:
    def __init__(self) -> None:
        self.layers: List[Dense_Layer] = []

    def add(self, layer=Dense_Layer()):
        self.layers.append(layer)
    def build(self, input_length=0):
        self.layers[0].build(input_length)

        for _ in range(1, len(self.layers)):
            self.layers[_].build(self.layers[_ - 1].no_of_neurons)

    def forward_pass(self, input_data=[]):




