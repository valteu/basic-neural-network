import numpy as np

import json

from Network import Network
import Layer


LR = 0.001
EPOCHS = 20

def get_w_b(filename):
    weights = []
    biases = []

    with open(filename, "r") as json_file:
        data = json.load(json_file)

    for item in data:
        if item["type"] == "w":
            weights.append(item["data"])
        elif item["type"] == "b":
            biases.append(item["data"])
    return weights, biases

def get_1d_data():
    return np.random.rand(int(10e3))

def get_1d_targets(d):
    return np.multiply(d, d)

def get_2d_data():
    return np.random.rand(int(10e4), 2)

def get_2d_targets(d):
    return np.sum(d, axis=1)

def get_xor_data():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def get_xor_targets(d):
    t = []
    for i in d:
        if i[0] == i[1]:
            t.append([0])
        else:
            t.append([1])
    return np.array(t)

def main():
    w, b = get_w_b("w_b.json")
    # data = get_1d_data()
    # targets = get_1d_targets(data)
    # data = get_2d_data()
    # targets = get_2d_targets(data)
    # data = get_xor_data()
    # targets = get_xor_targets(data)
    layers = [
        Layer.ReLu(1, 12),
        Layer.ReLu(12, 12),
        Layer.Sigmoid(12, 1)
    ]
    n = Network(layers, weights=w[0], biases=b[0])
    # n.train(EPOCHS, data, targets, LR)
    tests = np.array([0, 1, 0.3, 0.74565])
    t_targets = get_1d_targets(tests)
    n.test(tests, t_targets)
    # n.write("w_b.json")

if __name__ == "__main__":
    main()

