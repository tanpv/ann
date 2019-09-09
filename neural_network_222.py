'''
	Use the most simple form of neuron network

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
	- layer
		- weight from one layer to other layer
		- bias on one layer
	- multiple output
	- back propagation error
	- minimum neural network
'''

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
	def __init__(self, 
				network_structure):
		
		self.network_structure = network_structure
		
		self.input_num = self.network_structure[0]
		self.hiden_num = self.network_structure[1]
		self.output_num = self.network_structure[2]

		self.init_weight()
		self.init_bias()
		self.init_input_output()


	def init_weight(self):
		print('init weight')
		print('\n')
		self.w_input_to_hiden = np.random.randn(self.input_num, self.hiden_num)
		self.w_hiden_to_output = np.random.randn(self.hiden_num, self.output_num)
		print('w_input_to_hiden', self.w_input_to_hiden)
		print('\n')
		print('w_hiden_to_output', self.w_hiden_to_output)
		print('\n')


	def init_bias(self):
		print('init bias')
		self.b_hiden = np.random.randn(1, self.hiden_num)
		print('b_hiden', self.b_hiden)
		print('\n')


	def init_input_output(self):
		print('init input')
		self.input = np.array([[1,2]])
		print('input', self.input)
		self.output = np.array([[3,4]])
		print('output', self.output)
		print('\n')


	def activation(self, z):
		'''
			use sigmoid activation function
		'''
		return 1 / (1 + np.exp(-z))


	def z(self):
		'''
			sum(i*w) + b
		'''
		pass


	def loss(self):
		loss = np.square(self.feed_forward()-self.output)
		print('loss')
		print(loss)
		return loss


	def mse(self):
		mse = np.mean(np.square(self.feed_forward()-self.output))/2
		return mse


	def loss_derivative_respect_to_weight():
		pass


	def loss_derivative_respect_to_bias():
		pass


	def feed_forward(self):
		print('feed forward')
		print(self.input.shape)
		print(self.w_input_to_hiden.shape)
		print(self.b_hiden.shape)

		z = np.dot(self.input, self.w_input_to_hiden)+self.b_hiden
		print('z', z)

		s = self.sigmoid(z)
		print('s', s)

		o = np.dot(s, self.w_hiden_to_output)
		print('o', o)

		return o


network_structure = [2,2,2]

nn = NeuralNetwork(network_structure)
nn.feed_forward()
nn.loss()