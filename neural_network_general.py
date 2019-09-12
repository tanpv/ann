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

	- convert all of this to general network
'''


import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

	def __init__(self, 
				network,
				learning_rate,
				epoch):
		
		self.network = network
		self.learning_rate = learning_rate
		self.epoch = epoch

		self.init_weight()
		self.init_bias()
		self.init_input_output()


	def init_weight(self):
		print('init weight')
		self.weights = []
		for i in range(len(network)-1):
			weight = np.random.randn(self.network[i], self.network[i+1])
			print(weight.shape)
			self.weights.append(weight)


	def init_bias(self):
		print('init bias')
		self.biases = []
		for i in range(1, len(network)):
			bias = np.random.randn(self.network[i],1)
			print(bias.shape)
			self.biases.append(bias)


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
		self.e = np.square( self.a_nn[len(self.a_nn)-1]-self.output )/2
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
		delta_l1 = np.dot(weight_l1_to_l2, delta_l2)*a_l1*(1-a_l1)
		return delta_l1


	def delta_network(self, log=False):
		
		self.deltas = []

		# calculate delta at final layer
		a_output = self.a_nn[len(self.a_nn)-1]
		delta_output = (a_output - self.output)*a_output*(1-a_output)
		self.deltas.append(delta_output)

		# calculate delta of hiden layer
		delta_l2 = delta_output
		for i in range(len(self.network)-2):
			idx = len(self.network) - 2 - i
			
			a_l1 = self.a_nn[idx-1]
			weight_l1_to_l2 = self.weights[idx]
			delta_l2 = self.delta_layer(a_l1, weight_l1_to_l2, delta_l2)

			self.deltas.append(delta_l2)

		if log:
			for i in self.deltas:
				print('delta', i)
		

	def derivative(self, log=False):
		self.derivative_weights = []
		self.derivative_biases = []

		derivative_weight = self.deltas[len(self.a_nn)-1] * self.input.transpose()
		derivative_bias = self.deltas[len(self.a_nn)-1]

		for i in range(1, len(self.a_nn)):
			# broadcast
			derivative_weight = self.deltas[len(self.a_nn)-1-i] * self.a_nn[i].transpose()
			derivative_bias = self.deltas[len(self.weights)-1-i]

			self.derivative_weights.append(derivative_weight)
			self.derivative_biases.append(derivative_bias)

		# print(len(self.derivative_weights))
		# print(self.derivative_weights)


	def update_weight_bias(self):
		# update

		for i in range(len(self.weights)-1):
			print(self.weights[i])
			print(self.derivative_weights[i])
			self.weights[i] = self.weights[i] - self.learning_rate*self.derivative_weights[i]
			self.biases[i] = self.biases[i] - self.learning_rate*self.derivative_biases[i]


	def train(self):
		for i in range(self.epoch):
			self.feed_forward()
			self.delta_network()
			self.derivative()
			self.update_weight_bias()
			self.error()
			print('error at epoch {0} {1}'.format(i, self.e_sum))


	def feed_forward(self, log=False):
		input = self.input
		self.z_nn = []
		self.a_nn = []

		for i in range(len(self.network)-1):
		
			# calculate z then add to list
			z = self.z(input, self.weights[i], self.biases[i])
			self.z_nn.append(z)

			# calculate a then add to list
			a = self.a(z)
			self.a_nn.append(a)

			# use for next layor
			input = a

			if log:
				print('z', z)
				print('z.shape', z.shape)
				print('\n')
				print('a', a)
				print('a.shape', a.shape)
				print('\n')



network = [2,3,2]
learning_rate = 0.5
epoch = 500

nn = NeuralNetwork( network,
					learning_rate,
					epoch)

# nn.feed_forward(log=True)
# nn.delta_network(log=True)
# nn.derivative()
# nn.update_weight_bias()
# nn.error()
nn.train()