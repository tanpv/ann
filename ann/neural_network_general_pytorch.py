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

	- findout the best hyper parameter
		- structure
		- number of neuron on each layer
		- learning rate
		- epoch
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class NeuralNetwork():

	def __init__(self,
				network,
				learning_rate,
				epoch,
				load_model,
				gpu):
		
		self.gpu = gpu
		self.network = network
		self.learning_rate = learning_rate
		self.epoch = epoch

		# self.load_mnist()
		self.load_iris()
		
		if load_model:
			self.load_model()
		else:
			self.init_weight()
			self.init_bias()


	def load_iris(self):
		'''
			- load data from csv file
			- build input array and output array
		'''
		iris_frame = pd.read_csv('iris.csv')
		
		# convert species from text to number
		iris_frame.loc[iris_frame['species']=='setosa', 'species'] = 0
		iris_frame.loc[iris_frame['species']=='versicolor', 'species'] = 1
		iris_frame.loc[iris_frame['species']=='virginica', 'species'] = 2

		# normalize so value is inside 0,1
		max_for_normalize = iris_frame[['sepal_length',
										'sepal_width',
										'petal_length',
										'petal_width']].values.max()		
		
		output_size = 3
		if self.gpu:
			self.input = [torch.reshape(i, (4,1)) / max_for_normalize for i in torch.from_numpy(iris_frame[['sepal_length',
																						'sepal_width',
																						'petal_length',
																						'petal_width']].values).float().cuda('cuda:0')]

			self.output = [self.vectorized(j=o, output_size=output_size) for o in torch.from_numpy(iris_frame['species'].values).float().cuda('cuda:0')]

		else:
			self.input = [torch.reshape(i, (4,1)) / max_for_normalize for i in torch.from_numpy(iris_frame[['sepal_length',
																						'sepal_width',
																						'petal_length',
																						'petal_width']].values).float()]

			self.output = [self.vectorized(j=o, output_size=output_size) for o in torch.from_numpy(iris_frame['species'].values).float()]
		
		print('input.shape', self.input[0].shape)
		print('input', self.input[0])
		print('\n')
		print('output.shape', self.output[0].shape)
		print('output', self.output[0])
		print('\n')
		print('data length', len(self.input))

	def load_mnist(self):
		mnist_frame = pd.read_csv('mnist_train.csv')
		output_size = 10
		if self.gpu:
			self.input = [torch.reshape(i, (784,1))/255 for i in torch.from_numpy(mnist_frame.iloc[:,1:].values).float().cuda('cuda:0')]
			self.output = [self.vectorized(j=o, output_size=output_size) for o in torch.from_numpy(mnist_frame.iloc[:,0].values).float().cuda('cuda:0')]
		else:
			self.input = [torch.reshape(i, (784,1))/255 for i in torch.from_numpy(mnist_frame.iloc[:,1:].values).float()]
			self.output = [self.vectorized(j=o, output_size=output_size) for o in torch.from_numpy(mnist_frame.iloc[:,0].values).float()]

	def vectorized(self, j, output_size):
		if self.gpu:
			e = torch.zeros((output_size,1), dtype=torch.float).cuda('cuda:0')
		else:
			e = torch.zeros((output_size,1), dtype=torch.float)
		e[int(j)] = 1.
		return e


	def init_weight(self):
		print('init weight ---------------- ')
		self.weights = []
		for i in range(len(network)-1):
			# weight matrix with
			# with rows are number of neuron on (l-1) layer
			# with columns are number of neuron on l layer
			# note that for network of 3 layers --> have only 2 weight matrix
			# weight = np.random.randn(self.network[i], self.network[i+1])
			if self.gpu:
				weight = torch.randn((self.network[i], self.network[i+1])).float().cuda('cuda:0')
			else:
				weight = torch.randn((self.network[i], self.network[i+1])).float()
			
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
			# bias = np.random.randn(self.network[i],1)
			if self.gpu:
				bias = torch.randn(self.network[i], 1).float().cuda('cuda:0')
			else:
				bias = torch.randn(self.network[i], 1).float()

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
		# s = 1 / (1 + np.exp(-z))
		s = 1 / (1 + torch.exp(-z))
		if derivative:
			return s*(1-s)
		else:
			return s


	def z(self, input, weight, bias):
		'''
			sum(i*w) + b
		'''		
		z = torch.matmul(torch.t(weight), input) + bias
		return z


	def error(self, input, output):
		'''
			use mean square error
		'''
		self.feed_forward(input)
		diff = self.a_nn[len(self.a_nn)-1]-output
		self.e = diff**2/2
		self.e_sum = torch.sum(self.e)


	def delta_layer(self, a_l1, weight_l1_to_l2, delta_l2):
		'''
			suppose know
				- a at layer l1
				- weight from l1 to l2
				- delta at layer l2

			calculate
				- delta at layer l1
		'''
		# print('weight_l1_to_l2.shape', weight_l1_to_l2.shape)
		# print('delta_l2.shape', delta_l2.shape)
		# print('a_l1.shape', a_l1.shape)
		delta_l1 = torch.matmul(weight_l1_to_l2, delta_l2)*a_l1*(1-a_l1)

		return delta_l1


	def delta_network(self, output, log=False):
		
		self.deltas = []

		# calculate delta at final layer
		a_output = self.a_nn[len(self.a_nn)-1]
		delta_output = (a_output - output)*a_output*(1-a_output)
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
		

	def derivative(self, input, log=False):
		self.d_weights = []
		self.d_biases = []

		# print('self.deltas[len(self.a_nn)-1]', self.deltas[len(self.a_nn)-1])
		# print('self.deltas[len(self.a_nn)-1].shape', self.deltas[len(self.a_nn)-1].shape)
		# print('self.input.transpose().shape', self.input.transpose().shape)
		# print('\n')
		d_weight = self.deltas[len(self.a_nn)-1] * input.t()
		d_bias = self.deltas[len(self.a_nn)-1]
		self.d_weights.append(d_weight)
		self.d_biases.append(d_bias)

		for i in range(1, len(self.a_nn)):
			# broadcast
			# print('self.deltas[len(self.a_nn)-1-i].shape', self.deltas[len(self.a_nn)-1-i].shape)
			# print('self.a_nn[i].transpose().shape', self.a_nn[i].transpose().shape)
			d_weight = self.deltas[len(self.a_nn)-1-i] * self.a_nn[i].t()
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
			self.weights[i] = self.weights[i] - self.learning_rate*self.d_weights[i].t()
			self.biases[i] = self.biases[i] - self.learning_rate*self.d_biases[i]


	def train(self, log=False):
		self.errors = []

		for i in range(self.epoch):
			e_epoch = 0
			for idx, input in enumerate(self.input):
				output = self.output[idx]
				self.feed_forward(input, log)
				self.delta_network(output, log)
				self.derivative(input, log)
				self.update_weight_bias(log)
				self.error(input, output)
				e_epoch = e_epoch + self.e_sum
				# self.errors.append(self.e_sum)
			e_epoch = e_epoch / len(self.input)
			self.errors.append(e_epoch)
			print('error at epoch {0} {1}'.format(i, e_epoch))

		self.plot_train_error()
		self.predic()


	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


	def feed_forward(self, input, log=False):		
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

	def predic(self):
		idxs = [33, 44, 2, 10, 12, 60]

		for idx in idxs:
			input = self.input[idx]
			output = self.output[idx]
			self.feed_forward(input)
			print(self.a_nn[len(self.a_nn)-1])
			print(output)
			print('\n')

	def save_model(self):		
		np.save('weight', self.weights)
		np.save('bias', self.biases)

	def load_model(self):
		self.weights = np.load('weight.npy')
		self.biases = np.load('bias.npy')



# iris
# higher at middle is better
# # a good model for iris
network = [4,100,3]
network = [4,50,50,3]
network = [4,25,50,25,3]
network = [4,12,6,3]
learning_rate = 0.3
epoch = 1000
load_model = False
gpu = False

# mnist
# network = [784,40,10]
# learning_rate = 0.12
# epoch = 10
# load_model = False
# gpu = False

nn = NeuralNetwork( network,
					learning_rate,
					epoch,
					load_model,
					gpu)

print(torch.cuda.get_device_name('cuda:0'))

# nn.feed_forward(log=True, input=input)
# nn.delta_network(log=True, output = output)
# nn.derivative(log=True, input=input)
# nn.update_weight_bias()
# nn.error(input=input, output=output)
nn.train(log=False)
# nn.save_model()
