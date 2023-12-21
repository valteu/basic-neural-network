import numpy as np

import json
import matplotlib.pyplot as plt

from Network import Network
import Layer


LR = 0.001
EPOCHS = 50

def get_data():
    return np.random.rand(int(10e3))

def get_targets(d):
    return np.multiply(d, d)

def main():
    data = get_data()
    targets = get_targets(data)

    layers = [
        Layer.ReLu(1, 12),
        Layer.ReLu(12, 12),
        Layer.Sigmoid(12, 1)
    ]

    n = Network(layers)
    # uncomment and comment the line of code here to choose the gradient descent optimized training
    # n.train(EPOCHS, data, targets, LR)
    n.train_random(EPOCHS, data, targets, LR)

    # plotting loss
    loss = n.get_loss()
    loss = np.array(loss)
    epochs = np.arange(len(loss))
    plt.scatter(epochs, loss)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Cost value')
    plt.title('Neural Network cost over episodes')
    plt.show()

    # testing

    while True:
        tests = np.array([float(input("Test: "))])
        print(tests)
        print(n.test(tests)[0])

if __name__ == "__main__":
    main()

