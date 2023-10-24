import numpy as np
from Network import Network
import Layer

LR = 0.01
EPOCHS = 200

def get_data():
    return np.random.rand(int(10e3))

def get_xor_data():
    return np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])

def get_targets(d):
    return np.multiply(d, d)

def get_xor_targets(d):
    t = []
    for i in d:
        print(i[0], i[1])
        if i[0] == i[i]:
            t.append(0)
        else:
            t.append(1)
    return t

def main():
    data = get_data()
    targets = get_targets(data)
    # data = get_xor_data()
    # targets = get_xor_targets(data)
    layers = [
        Layer.ReLu(1, 12),
        Layer.ReLu(12, 12),
        Layer.Sigmoid(12, 1)
    ]
    l = len(layers)
    n = Network(layers, l)
    n.train(EPOCHS, data, targets, LR)
    tests = np.array([0, 0.1, 0.4, 0.2, 0.7])
    t_targets = get_targets(tests)
    n.test(tests, t_targets)

if __name__ == "__main__":
    main()
