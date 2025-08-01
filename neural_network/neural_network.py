import json

import numpy as np

from layer.layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.loss_list = []

    def add_layer(self, num_neurons, input_size):
        if not self.layers:
            self.layers.append(Layer(num_neurons, input_size))
        else:
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, x, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(x)):
                output = self.forward(x[i])
                loss += np.mean((y[i] - output) ** 2)
                loss_gradient = 2 * (output - y[i])
                self.backward(loss_gradient, learning_rate)
            loss /= len(x)
            self.loss_list.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, loss: {loss}")
    
    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            predictions.append(self.forward(x[i]))
        return np.array(predictions)
    
    def save(self, filename="model.json"):
        model_data = {
            "layers": []
        }

        for layer in self.layers:
            layer_data = {
                "num_neuron": len(layer.neurons),
                "input_size": len(layer.neurons[0].weight) if layer.neurons else 0,
                "neurons": layer.to_dict()
            }

            model_data["layers"].append(layer_data)

        with open(filename, "w") as f:
            json.dump(model_data, f)
        print(f"Model saved in {filename}")

    def load(self, filename="model.json"):
        with open(filename, "r") as f:
            model_data = json.load(f)

        self.layers = []

        for layer_data in model_data["layers"]:
            num_neurons = layer_data["num_neuron"]
            input_size = layer_data["input_size"]

            new_layer = Layer(num_neurons, input_size)

            new_layer.from_dict(layer_data["neurons"])

            self.layers.append(new_layer)

        print(f"Model loaded from {filename}")



    

if __name__ == "__main__":
    x = np.array([[0.5, 0.2, 0.1],
                   [0.9, 0.7, 0.3],
                   [0.4, 0.5, 0.8]])
    y = np.array([0.3, 0.6, 0.9])   #Data that neural network has to predict

    nn = NeuralNetwork()
    
    """"
    # Block to train the network
    nn.add_layer(num_neurons = 3, input_size = 3)
    nn.add_layer(num_neurons = 3, input_size = 3)
    nn.add_layer(num_neurons = 1, input_size = 4)

    
    nn.train(x, y, epochs = 10000, learning_rate = 0.1)

    nn.save("model1.json")
    # Block to train the network
    """

    #Line to load previous training
    nn.load("model1.json")
    
    predictions = nn.predict(x)
    print(f"Predictions: {predictions}")

    # python -m neural_network.neural_network
