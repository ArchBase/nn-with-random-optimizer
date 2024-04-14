class Neuron:
    def __init__(self, no_of_weights) -> None:
        self.weights = []
        self.bias = 0.0
        self.buffer = 0
        for _ in range(no_of_weights):
            self.weights.append(0.0)
    def forward_pass(self, input_data=[]):
        self.buffer = 0
        if len(input_data) != len(self.weights):
            print("Shape error")
            return None
        else:
            for _ in range(len(input_data)):
                self.buffer += self.weights[_] * input_data[_]
            self.buffer += self.bias
            self.buffer = max(0, self.buffer)
            return self.buffer


class Dense_Layer:
    def __init__(self, no_of_neurons, no_of_neurons_previous) -> None:
        self.no_of_neurons_previous = no_of_neurons_previous
        self.no_of_neurons = no_of_neurons
        self.neurons = []
        self.buffer = []
        for _ in range(self.no_of_neurons):
            self.neurons.append(Neuron(no_of_weights=self.no_of_neurons_previous))
    def forward_pass(self, input_data=[]):
        self.buffer.clear()
        for _ in range(self.no_of_neurons):
            self.buffer.append(self.neurons[_].forward_pass(input_data=input_data))
        
         


class Input_Layer(Dense_Layer):
    def __init__(self, no_of_neurons) -> None:
        super().__init__(no_of_neurons=no_of_neurons, no_of_neurons_previous=1)
    def forward_pass(self, input_data=[]):
        self.buffer.clear()
        for _ in range(len(input_data)):
            self.buffer.append(self.neurons[_].forward_pass(input_data[_]))

        return self.buffer


class Sequential_Neural_Network:
    def __init__(self) -> None:
        self.layers = []
    def add(self, no_of_neurons):
        if not self.layers:
            self.layers.append(Input_Layer(no_of_neurons=no_of_neurons))
        else:
            self.layers.append(Dense_Layer(no_of_neurons=no_of_neurons, no_of_neurons_previous=self.layers[-1].no_of_neurons))