import numpy as np

from neuron.neuron import Neuron


class Layer:

    def __init__(self, num_neurons, inputs_size):
        self.neurons = [Neuron(inputs_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, d_outputs, learning_rate):
        d_inputs = np.zeros(len(self.neurons[0].input))
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs
    
    def to_dict(self):
        return [neuron.to_dict() for neuron in self.neurons]

    def from_dict(self, data):
        for neuron, neuron_data in zip(self.neurons, data):
            neuron.from_dict(neuron_data)
    

if __name__ == "__main__":
    layer = Layer(3, 4)
    inputs = np.array([1, 8, 5, 6])

    layer_output = layer.forward(inputs)
    print("Layer outputs:", layer_output)

    # python -m layer.layer
        
