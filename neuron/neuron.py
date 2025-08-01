import numpy as np

class Neuron:

    def __init__(self, n_input):
        self.weight = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.input = None
        self.dweight = np.zeros_like(self.weight)
        self.dbias = 0

    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivate_activate(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        self.input = inputs
        weighted_sum = np.dot(inputs, self.weight) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output
    
    def backward(self, d_output, learning_rate):
        d_activation = d_output * self.derivate_activate(self.output)
        self.dweight = np.dot(self.input, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weight)
        self.weight -= self.dweight * learning_rate
        self.bias -= learning_rate * self.bias
        return d_input
    
    def to_dict(self):
        return {
            "weights": self.weight.tolist(),
            "bias": self.bias
        }

    def from_dict(self, data):
        self.weight = np.array(data["weights"])
        self.bias = data["bias"]


if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1, 2, 3])
    output = neuron.forward(inputs)

    print("Neuron output:", output)
