'''

	- define network structure with a list
		- number of layer
		- number of each neural in each layer

	- init weight and bias with matrix shape

	- feed forward
		- calculate z
		- calculate a
		- to final layer

	- back propagation
		- theory on calculate derivative output layer
		- theory on calculate derivative hiden layer
		- theory on calculate delta output layer
		- theory on calculate delta hiden layer
		- code calculate delta output
		- code calculate delta hiden layer

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
		print('init weight --------------------')
		self.w_input_to_hiden = np.random.randn(self.input_num, self.hiden_num)
		self.w_hiden_to_output = np.random.randn(self.hiden_num, self.output_num)
		
		print('w_input_to_hiden.shape')
		print(self.w_input_to_hiden.shape)
		
		print('w_input_to_hiden')
		print(self.w_input_to_hiden)
		print('\n')
		
		print('w_hiden_to_output.shape')
		print(self.w_hiden_to_output.shape)

		print('w_hiden_to_output')
		print(self.w_hiden_to_output)
		print('\n')


	def init_bias(self):
		print('init bias -------------------- ')
		self.b_hiden = np.random.randn(self.hiden_num, 1)
		print('b_hiden.shape')
		print(self.b_hiden.shape)
		print('b_hiden')
		print(self.b_hiden)

		self.b_output = np.random.randn(self.output_num, 1)
		print('b_output.shape')
		print(self.b_output.shape)
		print('b_output')
		print(self.b_output)
		print('\n')


	def init_input_output(self):
		print('init input')
		self.input = np.array([[0.1, 0.3]]).reshape(2,1)
		print('input')
		print(self.input)
		self.output = np.array([[0.7, 0.2]]).reshape(2,1)
		print('output')
		print(self.output)
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


	def delta_layer(self, a_l1, weight_l1_to_l2, delta_l2):
		'''
			suppose know
				- a at layer l1
				- weight from l1 to l2
				- delta at layer l2

			calculate
				- delta at layer l1
		'''
		# formular to calculate delta
		delta_l1 = np.dot(weight_l1_to_l2, delta_l2)*a_l1*(1-a_l1)
		return delta_l1


	def delta_network(self, log=False):
		print('delta ----------------------')
		self.delta_output = (self.a_output-self.output)*self.a_output*(1-self.a_output)
		if log:
			print('delta_output.shape')
			print(self.delta_output.shape)
			print('delta_output')
			print(self.delta_output)
			print('\n')

		# need reshape after dot product here
		# self.delta_hiden = np.dot(self.w_hiden_to_output, self.delta_output) * self.a_hiden * (1-self.a_hiden)
		self.delta_hiden = self.delta_layer(self.a_hiden, self.w_hiden_to_output, self.delta_output)
		if log:
			print('delta_hiden.shape')
			print(self.delta_hiden.shape)
			print('delta_hiden')
			print(self.delta_hiden)
			print('\n')


	def derivative(self, log=False):
		print('self.a_hiden.shape')
		print(self.a_hiden.shape)

		print('self.delta_output.shape')
		print(self.delta_output.shape)

		self.derivative_error_respect_w_out = self.a_hiden * self.delta_output.transpose()
		if log:
			print('derivative weight out shape')
			print(self.derivative_error_respect_w_out.shape)
			print('derivative weight out')
			print(self.derivative_error_respect_w_out)
			print('\n')

		self.derivative_error_respect_b_out = self.delta_output
		if log:
			print('derivative bias out shape')
			print(self.derivative_error_respect_b_out.shape)
			print('derivative bias out')
			print(self.derivative_error_respect_b_out)
			print('\n')

		self.derivative_error_respect_w_hiden = self.delta_hiden * self.input.transpose()
		if log:
			print('derivative weight hiden shape')
			print(self.derivative_error_respect_w_hiden.shape)
			print('derivative weight hiden')
			print(self.derivative_error_respect_w_hiden)
			print('\n')

		self.derivative_error_respect_b_hiden = self.delta_hiden
		if log:
			print('derivative bias hiden shape')
			print(self.derivative_error_respect_b_hiden.shape)
			print('derivative bias hiden')
			print(self.derivative_error_respect_b_hiden)
			print('\n')


	def update_weight_bias(self):
		# update
		self.w_input_to_hiden = self.w_input_to_hiden - self.learning_rate*self.derivative_error_respect_w_hiden.transpose()
		self.b_hiden = self.b_hiden - self.learning_rate*self.derivative_error_respect_b_hiden
		self.w_hiden_to_output = self.w_hiden_to_output - self.learning_rate*self.derivative_error_respect_w_out
		self.b_output = self.b_output - self.learning_rate*self.derivative_error_respect_b_out


	def train(self):
		self.errors = []
		for i in range(self.epoch):
			self.feed_forward()
			self.delta_network()
			self.derivative()
			self.update_weight_bias()
			self.error()
			self.errors.append(self.e_sum)
			print('error at epoch {0} {1}'.format(i, self.e_sum))

		self.plot_train_error()


	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


	def feed_forward(self, log=False):
		if log:
			print('feed forward ------------------')

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


network_structure = [2,3,2]
learning_rate = 0.1
epoch = 300

nn = NeuralNetwork(network_structure,
					learning_rate,
					epoch)

nn.feed_forward(log=True)
nn.delta_network(log=True)
nn.derivative(log=True)
# nn.error()
# nn.train()