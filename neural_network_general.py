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
		print('init weight ---------------- ')
		self.weights = []
		for i in range(len(network)-1):
			# weight matrix with
			# with rows are number of neuron on (l-1) layer
			# with columns are number of neuron on l layer
			# note that for network of 3 layers --> have only 2 weight matrix
			weight = np.random.randn(self.network[i], self.network[i+1])			
			self.weights.append(weight)
			print('weight.shape' ,weight.shape)
			print('weight', weight)
			print('\n')


	def init_bias(self):
		print('init bias ---------------- ')
		self.biases = []
		for i in range(1, len(network)):
			# input layer no have bias
			# network with l layers have l-1 bias matrix
			# number of rows equal to number of neuron on one layer
			bias = np.random.randn(self.network[i],1)
			self.biases.append(bias)
			print('bias.shape', bias.shape)
			print('bias', bias)
			print('\n')


	def init_input_output(self):
		print('init input output -------------------')
		# input is init as matrix
		# 1 column and number of row is number of input
		# same with output
		self.input = np.array([[0.8, 0.1]]).reshape(2,1)
		print('input', self.input)
		self.output = np.array([[0.4, 0.7]]).reshape(2,1)
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
			print('delta -----------------------')
			for i in self.deltas:
				print('delta.shape', i.shape)
				print('delta', i)
				print('\n')
		

	def derivative(self, log=False):
		self.d_weights = []
		self.d_biases = []

		# print('self.deltas[len(self.a_nn)-1]', self.deltas[len(self.a_nn)-1])
		# print('self.deltas[len(self.a_nn)-1].shape', self.deltas[len(self.a_nn)-1].shape)
		# print('self.input.transpose().shape', self.input.transpose().shape)
		# print('\n')
		d_weight = self.deltas[len(self.a_nn)-1] * self.input.transpose()
		d_bias = self.deltas[len(self.a_nn)-1]
		self.d_weights.append(d_weight)
		self.d_biases.append(d_bias)

		for i in range(1, len(self.a_nn)):
			# broadcast
			# print('self.deltas[len(self.a_nn)-1-i].shape', self.deltas[len(self.a_nn)-1-i].shape)
			# print('self.a_nn[i].transpose().shape', self.a_nn[i].transpose().shape)
			d_weight = self.deltas[len(self.a_nn)-1-i] * self.a_nn[i].transpose()
			d_bias = self.deltas[len(self.a_nn)-1-i]

			self.d_weights.append(d_weight)
			self.d_biases.append(d_bias)

		if log:
			print('derivative weights ----------------------')
			for d in self.d_weights:
				print('d_weight.shape', d.shape)
				print('d_weight', d)
				print('\n')

			for d in self.d_biases:
				print('d_bias.shape', d.shape)
				print('d_bias', d)
				print('\n')

	def update_weight_bias(self, log=False):
		# update

		for i in range(len(self.weights)-1):
			# print('self.weights[i].shape', self.weights[i].shape)
			# print('self.d_weights[i].shape', self.d_weights[i].shape)
			self.weights[i] = self.weights[i] - self.learning_rate*self.d_weights[i].transpose()
			self.biases[i] = self.biases[i] - self.learning_rate*self.d_biases[i]


	def train(self, log=False):
		self.errors = []		
		for i in range(self.epoch):
			self.feed_forward(log)
			self.delta_network(log)
			self.derivative(log)
			self.update_weight_bias(log)
			self.error()
			self.errors.append(self.e_sum)
			print('error at epoch {0} {1}'.format(i, self.e_sum))

		self.plot_train_error()


	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


	def feed_forward(self, log=False):
		input = self.input
		self.z_nn = []
		self.a_nn = []

		for i in range(len(self.network)-1):
		
			# calculate z then add to list
			z = self.z(input, self.weights[i], self.biases[i])
			# print('z.shape', z.shape)
			self.z_nn.append(z)

			# calculate a then add to list
			a = self.a(z)
			# print('a.shape', a.shape)
			self.a_nn.append(a)

			# use for next layor
			input = a

		if log:
			print('feed forward a -------------------')
			print(self.a_nn[0])
			print(self.a_nn[1])
			# for i in range(len(self.a_nn)):
			# 	print('a', self.a_nn[i])
			# 	print('a.shape', self.a_nn[i])
			# 	print('\n')

			print('feed forward z -------------------')
			print(self.z_nn[0])
			print(self.z_nn[1])
			# for i in self.z_nn:				
			# 	print('z', z)
			# 	print('z.shape', z.shape)
			# 	print('\n')



network = [2,3,2]
learning_rate = 10
epoch = 1000

nn = NeuralNetwork( network,
					learning_rate,
					epoch)

# nn.feed_forward(log=True)
# nn.delta_network(log=True)
# nn.derivative()
# nn.update_weight_bias()
# nn.error()
nn.train(log=False)