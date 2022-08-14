import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 0.001
SAMPLESIZE = 20

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss

class Loss_CategoricalCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

def visualize_nnn(dense1, X):
	"""
	spiral data (300, 3)
	0 = unten-links
	1 = oben-rechts
	2 = oben-links
	"""
	plt.scatter(dense1.weights[0], dense1.weights[1])
	X, y = spiral_data(100, 3)
	X_x = data_extract(X, 0)
	X_y = data_extract(X, 1)
	plt.scatter(X_x, X_y, c=y)
	plt.show()

	plt.show()

def tolerance(episode):
	return tolerance

def data_extract(X, index):
	return list(list(zip(*X))[index])
def train(epochs):
	X, y = spiral_data(100, 3)
	X_sampled = []
	list_ = []
	for i in range(int(X.size/2)):
		list_.append(X[i])
		if i%SAMPLESIZE == 0 and i != 0:
			X_sampled.append(list_)
			list_ = []
	y_sampled = []	
	list_ = []
	for i in range(int(y.size)):
		list_.append(y[i])
		if i%SAMPLESIZE == 0 and i != 0:
			list_array = np.array(list_)
			y_sampled.append(list_array)
			list_ = []
	y_sampled_array = np.array(y_sampled)
	#
	y_sampled_array = y
	X_sampled = X
	#
	dense1 = Layer_Dense(2, 20)
	activation1 = Activation_ReLU()

	dense2 = Layer_Dense(20, 20)
	activation2 = Activation_Softmax()

	dense3 = Layer_Dense(20, 3)
	activation3 = Activation_Softmax()

	loss_function = Loss_CategoricalCrossentropy()

	best_loss = 10
	best_weights_l1 = 0.1 * np.random.randn(2, 3)
	best_biases_l1 = 0.1 * np.random.randn(1, 3)
	best_weights_l2 = 0.1 * np.random.randn(3, 3)
	best_biases_l2 = 0.1 * np.random.randn(1, 3)
	best_weights_l3 = 0.1 * np.random.randn(3, 3)
	best_biases_l3 = 0.1 * np.random.randn(1, 3)

	episode_weights_l1 = 0.1 * np.random.randn(2, 3)
	episode_biases_l1 = 0.1 * np.random.randn(1, 3)
	episode_weights_l2 = 0.1 * np.random.randn(3, 3)
	episode_biases_l2 = 0.1 * np.random.randn(1, 3)
	episode_weights_l3 = 0.1 * np.random.randn(3, 3)
	episode_biases_l3 = 0.1 * np.random.randn(1, 3)
	activation1 = Activation_ReLU()
	activation2 = Activation_ReLU()
	activation3 = Activation_Softmax()
	# for samples in range(len(y_sampled)):
	# 	print(samples)
	for ii in range(epochs):

		for i in range(best_weights_l1.shape[0]):
			for ii in range(best_weights_l1.shape[1]):
				episode_weights_l1[i][ii] = best_weights_l1[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)
		for i in range(best_biases_l1.shape[0]):
			for ii in range(best_biases_l1.shape[1]):
				episode_biases_l1[i][ii] = best_biases_l1[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)
		for i in range(best_weights_l2.shape[0]):
			for ii in range(best_weights_l2.shape[1]):
				episode_weights_l2[i][ii] = best_weights_l2[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)
		for i in range(dense2.biases.shape[0]):
			for ii in range(best_biases_l2.shape[1]):
				episode_biases_l2[i][ii] = dense2.biases[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)
		for i in range(best_weights_l3.shape[0]):
			for ii in range(best_weights_l3.shape[1]):
				episode_weights_l3[i][ii] = best_weights_l3[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)
		for i in range(dense3.biases.shape[0]):
			for ii in range(best_biases_l3.shape[1]):
				episode_biases_l3[i][ii] = dense3.biases[i][ii] + random.uniform(-TOLERANCE, TOLERANCE)

		dense1.weights = episode_weights_l1
		dense1.biases = episode_biases_l1

		dense2.weights = episode_weights_l2
		dense2.biases = episode_biases_l2

		dense3.weights = episode_weights_l3
		dense3.biases = episode_biases_l3

		dense1.forward(X_sampled)
		activation1.forward(dense1.output)

		dense2.forward(activation1.output)
		activation2.forward(dense2.output)

		dense3.forward(activation2.output)
		activation3.forward(dense3.output)
		loss = loss_function.calculate(activation2.output, y_sampled_array)
		if loss < best_loss:
			best_loss = loss
			print("loss: ", loss)
			best_weights_l1 = episode_weights_l1
			best_biases_l1 = episode_biases_l1
			best_weights_l2 = episode_weights_l2
			best_biases_l2 = episode_biases_l2
	test = [0.535, 0.825]
	dense1.forward(X_sampled)
	activation1.forward(dense1.output)

	dense2.forward(activation1.output)
	activation2.forward(dense2.output)

	dense3.forward(activation2.output)
	activation3.forward(dense3.output)
	print("output: ", activation3.output)
	visualize_nnn(dense1, X)
train(epochs=1000000)
