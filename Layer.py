import numpy as np

import json
class Layer:

    def __init__(self, inp, out, weights=False, biases=False):
        self.i = inp # input size
        self.o = out # output size

        self.pre = np.zeros(self.o) # pre activation output
        self.out = np.zeros(self.o) # activations

        self.d = np.zeros(self.o) # delta
        if weights:
            self.w = weights
        else:
            self.w = np.random.rand(self.i, self.o) # weights

        if biases:
            self.b = biases
        else:
            self.b = np.random.rand(self.o) # biases

        self.g_w = np.zeros((self.i, self.o)) # gradient weights
        self.g_b = np.zeros(self.o) # gradient biases

        self.input_data = None

    def forward(self, d):
        self.input_data = d
        self.pre = np.dot(d, self.w) + self.b
        self.activation()
        return self.out

    def backward(self, gradient):
        self.d = gradient * self.derivative()

        for i in range(self.i):
            for j in range(self.o):
                self.g_w[i, j] = self.d[j] * self.input_data[i]
        self.g_b = np.sum(self.d, axis=0)

        gradient_to_pass = np.dot(self.d, self.w.T)
        return gradient_to_pass

    def activation(self):
        pass

    def derivative(self):
        pass

    def write(self, filename):
        data = []
        data.append("Weights:")
        for w in self.w:
            data.append(w.tolist())
        data.append("Biases")
        data.append(self.b.tolist())
        with open(filename, 'a') as json_file:
            json.dump(data, json_file)
            json_file.write('\n')

class Linear(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def activation(self):
        self.out = self.pre

    def derivative(self):
        return np.ones(self.o)


class ReLu(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def activation(self):
        self.out = np.maximum(self.pre, 0)

    def derivative(self):
        return np.where(self.out > 0, 1, 0)


class Sigmoid(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def sigmoid(self, d):
        return 1 / (1 + np.exp(-d))

    def activation(self):
        self.out = self.sigmoid(self.pre)

    def derivative(self):
        return self.sigmoid(self.out) * (1 - self.sigmoid(self.out))
