import numpy as np
class Layer:

    def __init__(self, inp, out):
        self.i = inp # input size
        self.o = out # output size

        self.out = np.zeros(self.o) # pre activation output
        self.a = np.zeros(self.o) # activations

        self.d = np.zeros(self.o) # delta
        self.w = np.random.rand(self.i, self.o) # weights
        self.b = np.random.rand(self.o) # biases

        self.g_w = np.zeros((self.i, self.o)) # gradient weights
        self.g_b = np.zeros(self.o) # gradient biases

        self.input_data = None

    def forward(self, d):
        self.input_data = d
        self.out = np.dot(d, self.w) + self.b
        self.activation()
        return self.a

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

class Linear(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def activation(self):
        self.a = self.out

    def derivative(self):
        return np.ones(self.o)


class ReLu(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def activation(self):
        self.a = np.maximum(self.out, 0)

    def derivative(self):
        return np.where(self.a > 0, 1, 0)


class Sigmoid(Layer):
    def __init__(self, inp, out):
        super().__init__(inp, out)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def activation(self):
        self.a = self.sigmoid(self.out)

    def derivative(self):
        return self.sigmoid(self.a) * (1 - self.sigmoid(self.a))
