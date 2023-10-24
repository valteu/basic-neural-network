import numpy as np

class Network:
    def __init__(self, layout, n_layers):
        self.n = n_layers
        self.layers = layout
        self.i = self.layers[0].i
        self.o = self.layers[self.n-1].o

        self.out = np.zeros(self.o)

    def forward(self, d):
        out = d
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient):
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, lr):
        for layer in self.layers:
            layer.w -= lr * layer.g_w
            layer.b -= lr * layer.g_b

    def squared_loss(self, pred, t):
        return (pred - t) ** 2

    def gradient_of_loss(self, predicted_output, target):
        # Compute the gradient of the mean squared error loss
        num_samples = len(target)
        return 2 * (predicted_output - target) / num_samples

    def train(self, epochs, data, targets, lr):
        for epoch in range(epochs):
            total_loss = 0
            for sample in range(len(data)):
                d_tmp = data[self.i * sample : self.i * (1 + sample)]
                t_tmp = targets[self.o * sample : self.o * (1 + sample)]
                pred_out = self.forward(d_tmp)
                loss = self.squared_loss(pred_out, t_tmp)
                total_loss += loss
                loss_gradient = self.gradient_of_loss(pred_out, t_tmp)
                self.backward(loss_gradient)
                self.update(lr)
            print(f"Loss: {total_loss/ len(data)}")

    def test(self, tests, targets):
        t = tests.reshape(-1, self.i)
        t_targets = targets.reshape(-1, self.o)

        for i in range(len(t)):
            self.forward(t[i])
            print(f"Tested: {t[i]}, recieved: {self.layers[-1].a}, target: {t_targets[i]}")
