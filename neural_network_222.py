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
				network_structure,
				learning_rate,
				epoch):
		
		self.network_structure = network_structure
		
		self.input_num = self.network_structure[0]
		self.hiden_num = self.network_structure[1]
		self.output_num = self.network_structure[2]

		self.learning_rate = learning_rate
		self.epoch = epoch

		self.init_weight()
		self.init_bias()
		self.init_input_output()


	def init_weight(self):
		print('init weight')
		print('\n')
		# what about number of row and column ?
		self.w_input_to_hiden = np.random.randn(self.input_num, self.hiden_num)
		self.w_hiden_to_output = np.random.randn(self.hiden_num, self.output_num)
		print('w_input_to_hiden', self.w_input_to_hiden)
		print('\n')
		print('w_hiden_to_output', self.w_hiden_to_output)
		print('\n')


	def init_bias(self):
		print('init bias')

		self.b_hiden = np.random.randn(self.hiden_num, 1)
		print('b_hiden', self.b_hiden)

		self.b_output = np.random.randn(self.output_num, 1)
		print('b_output', self.b_output)
		print('\n')


	def init_input_output(self):
		print('init input')
		self.input = np.array([[0.1, 0.3]]).reshape(2,1)
		print('input', self.input)
		self.output = np.array([[0.7, 0.2]]).reshape(2,1)
		print('output', self.output)
		print('\n')


	def a(self, z, derivative=False):
		'''
			use sigmoid activation function
		'''
		s = 1 / (1 + np.exp(-z))
		if derivative:
			return s*(1-s)
		else:
			return s


	def z(self, input, weight, bias):
		'''
			sum(i*w) + b
		'''
		# print('input.shape', input.shape)
		# print('weigh.shape', weight.shape)
		# print('bias.shape', bias.shape)
		z = np.dot(np.transpose(weight), input) + bias
		return z


	def error(self):
		'''
			use mean square error
		'''
		self.feed_forward()
		self.e = np.square(self.a_output-self.output)/2
		self.e_sum = np.sum(self.e)

		# print('e', self.e)
		# print('e_sum', self.e_sum)



	def delta(self, log=False):
		self.delta_output = (self.a_output-self.output)*self.a_output*(1-self.a_output)
		if log:
			print('delta_output', self.delta_output)
			print('delta_output.shape', self.delta_output.shape)
			print('\n')

		# need reshape after dot product here
		self.delta_hiden = np.dot(self.w_hiden_to_output, self.delta_output) * self.a_hiden * (1-self.a_hiden)
		if log:
			print('delta_hiden', self.delta_hiden)
			print('delta_hiden shape', self.delta_hiden.shape)
			print('\n')


	def derivative(self, log=False):
		self.derivative_error_respect_w_out = self.w_hiden_to_output * self.delta_output
		if log:
			print('derivative weight out', self.derivative_error_respect_w_out)
			print('derivative weight out shape', self.derivative_error_respect_w_out.shape)
			print('\n')

		self.derivative_error_respect_b_out = self.delta_output
		if log:
			print('derivative bias out', self.derivative_error_respect_b_out)
			print('derivative bias out shape', self.derivative_error_respect_b_out.shape)
			print('\n')

		self.derivative_error_respect_w_hiden = self.w_input_to_hiden * self.delta_hiden
		if log:
			print('derivative weight hiden', self.derivative_error_respect_w_hiden)
			print('derivative weight hiden shape', self.derivative_error_respect_w_hiden.shape)
			print('\n')

		self.derivative_error_respect_b_hiden = self.delta_hiden
		if log:
			print('derivative bias hiden', self.derivative_error_respect_b_hiden)
			print('derivative bias hiden shape', self.derivative_error_respect_b_hiden.shape)
			print('\n')


	def update_weight_bias(self):
		# update
		self.w_input_to_hiden = self.w_input_to_hiden - self.learning_rate*self.derivative_error_respect_w_hiden
		self.b_hiden = self.b_hiden - self.learning_rate*self.derivative_error_respect_b_hiden
		self.w_hiden_to_output = self.w_hiden_to_output - self.learning_rate*self.derivative_error_respect_w_out
		self.b_output = self.b_output - self.learning_rate*self.derivative_error_respect_b_out


	def train(self):
		for i in range(self.epoch):
			self.feed_forward()
			self.delta()
			self.derivative()
			self.update_weight_bias()
			self.error()
			print('error at epoch {0} {1}'.format(i, self.e_sum))


	def feed_forward(self, log=False):
		if log:
			print('feed forward')
			print(self.input.shape)
			print(self.w_input_to_hiden.shape)
			print(self.b_hiden.shape)
			print('\n')

		self.z_hiden = self.z(self.input, self.w_input_to_hiden, self.b_hiden)
		if log:
			print('z_hiden', self.z_hiden)
			print('z_hiden.shape', self.z_hiden.shape)
			print('\n')

		self.a_hiden = self.a(self.z_hiden)
		if log:
			print('a_hiden', self.a_hiden)
			print('a_hiden.shape', self.a_hiden.shape)
			print('\n')

		self.z_output = self.z(self.a_hiden, self.w_hiden_to_output, self.b_output)
		if log:
			print('z_output', self.z_output)
			print('z_output.shape', self.z_output.shape)
			print('\n')

		self.a_output = self.a(self.z_output)
		if log:
			print('a_output', self.a_output)
			print('a_output.shape', self.a_output.shape)
			print('\n')		



network_structure = [2,2,2]
learning_rate = 0.5
epoch = 100

nn = NeuralNetwork(network_structure,
					learning_rate,
					epoch)
# nn.feed_forward()
# nn.delta()	
# nn.derivative()
# nn.e()

nn.train()