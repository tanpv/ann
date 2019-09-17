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
	- train
	- learning rate
		- too big with value 1
		- too small with value 0.01
		- normal learning rate at 0.1
	- batch size
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
		# number of weight equal with number if input
		self.weight = np.random.randn(len(self.input))
		print('weight', self.weight)

	def init_bias(self):
		self.bias = np.random.randn()
		print('bias', self.bias)

	def input_weight_bias(self, input, weight, bias):
		sum = 0
		for i in range(len(self.input)):
			sum = sum + weight[i] * input[i]
		return sum+bias

	def feed_forward(self, input, weight, bias):
		return self.sigmoid(self.input_weight_bias(input, weight, bias))

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def loss(self, input, weight, bias, output):
		# caculate loss with mean square error
		# mse = square(expected_output - real_output)
		return np.square( self.feed_forward(input, weight, bias) - output)

	def loss_derivative_respect_to_weight(self,
										input,
										weight, 
										bias,
										output):

		z = self.input_weight_bias(input, weight, bias)
		# return a list of derivative
		return [2*(self.sigmoid(z) - output)*self.sigmoid(z)*(1-self.sigmoid(z))*self.input[i] for i in range(len(input))]

	def loss_derivative_respect_to_bias(self, 
										input, 
										weight, 
										bias,
										output):
		z = self.input_weight_bias(input, weight, bias)
		return 2*(self.sigmoid(z)-output)*self.sigmoid(z)*(1-self.sigmoid(z))

	def train(self):
		self.losses = []
		for n in range(self.epoch):

			# calculate current loss
			loss = self.loss(self.input, self.weight, self.bias, self.output)
			print('error at epoch {0} is {1}'.format(n, loss))
			self.losses.append(loss)


			w_derivative = self.loss_derivative_respect_to_weight(self.input, 
																self.weight,
																self.bias,
																self.output)
			# update weight and bias
			for i in range(len(self.input)):
				self.weight[i] = self.weight[i] - self.learning_rate*w_derivative[i]

			self.bias = self.bias - self.learning_rate*self.loss_derivative_respect_to_bias(self.input,
																							self.weight,
																							self.bias,
																							self.output)

			print('updated weight {0}'.format(self.weight))
			print('updated bias {0}'.format(self.bias))
			print('\n')

		self.plot_train_error()
  
	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.losses)
		plt.show()



input = [0.3, 0.1, 0.7, 0.11, 0.8, 0.22, 0.6, 0.9, 0.45, 0.5]

# note that output of sigmoid function is between (0,1)
output = 0.89

learning_rate = 0.1

epoch = 100

n = Neuron(	input,
			output,
			learning_rate,
			epoch )

n.train()
