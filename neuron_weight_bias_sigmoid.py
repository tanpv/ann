'''
	- multiple data samples
	- neuron
		- weight
		- bias
		- activation function
	- feed forward
	- mean square error
		- plot 3d for error
		- https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725
	- training
	- learning rate
		- too big with value 1
		- too small with value 0.01
		- normal learning rate at 0.1
	- epoch
'''

import numpy as np
import matplotlib.pyplot as plt

class Neuron(object):
	def __init__(self, 
				input,
				output,
				learning_rate,
				epoch):

		self.input = input
		self.output = output
		self.learning_rate = learning_rate
		self.epoch = epoch

		self.init_weight()
		self.init_bias()

	def init_weight(self):
		self.weight = np.random.randn()
		print('weight', self.weight)

	def init_bias(self):
		self.bias = np.random.randn()
		print('bias', self.bias)

	def feed_forward(self):
		z = self.z(self.input[0], self.weight, self.bias)
		return self.sigmoid(z)

	def z(self, input, weight, bias):
		# z = input * weight + bias
		z = input*weight + bias
		return z

	def sigmoid(self, z):
		return 1./(1. + np.exp(-z))

	def sigmoid_derivative_respect_to_z(self, z):
		return z*(1-z)

	def loss(self):
		# caculate loss with mean square error
		# mse = square(expected_output - real_output)
		return np.square( self.feed_forward() - self.output[0])

	def loss_derivative_respect_to_weight(self, weight, bias):
		z = self.input[0] * weight + bias
		return 2*(self.sigmoid(z)-self.output[0])*self.sigmoid(z)*(1-self.sigmoid(z))*self.input[0]

	def loss_derivative_respect_to_bias(self, weight, bias):
		z = self.input[0] * weight + bias
		return 2*(self.sigmoid(z)-self.output[0])*self.sigmoid(z)*(1-self.sigmoid(z))

	def gradient_b(self, b):
		return -2*self.sigmoid(z)*np.square(1-self.sigmoid(z))

	def train(self):
		self.losses = []
		for n in range(self.epoch):
			loss = self.loss()
			print('error at epoch {0} is {1}'.format(n, loss))
			self.losses.append(loss)

			self.weight = self.weight - self.learning_rate*self.loss_derivative_respect_to_weight(self.weight, self.bias)
			self.bias = self.bias - self.learning_rate*self.loss_derivative_respect_to_bias(self.weight, self.bias)

			print('updated weight {0}'.format(self.weight))
			print('updated bias {0}'.format(self.bias))
			print('\n')

		self.plot_train_error()
  
	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.losses)
		plt.show()


input = [1]
output = [2]

learning_rate = 0.3
epoch = 300

n = Neuron(	input,
			output,
			learning_rate,
			epoch )

n.train()


