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

	- stochastic gradient descent
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork():

	def __init__(self,
				network,
				learning_rate,
				epoch,
				load_model,
				batch_size):
		
		self.network = network
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.batch_size =batch_size

		self.load_data()
		
		self.shuffle_data_in_batch()
		
		if load_model:
			self.load_model()
		else:
			self.init_weight()
			self.init_bias()


	def load_data(self):
		'''
			- load data from csv file
			- build input array and output array
		'''
		self.iris_frame = pd.read_csv('iris.csv')
		
		# convert species from text to number
		self.iris_frame.loc[self.iris_frame['species']=='setosa', 'species'] = 0
		self.iris_frame.loc[self.iris_frame['species']=='versicolor', 'species'] = 1
		self.iris_frame.loc[self.iris_frame['species']=='virginica', 'species'] = 2

		# normalize so value is inside 0,1
		self.max_for_normalize = self.iris_frame[['sepal_length',
												'sepal_width',
												'petal_length',
												'petal_width']].values.max()

	def index_marks(self, nrows, chunk_size):
		return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

	def shuffle_data_in_batch(self):
		# random suffle
		self.iris_frame = self.iris_frame.sample(frac=1).reset_index(drop=True)

		batch_num = len(self.iris_frame) // self.batch_size
		
		# add to batchs
		self.batchs=[]
		start = 0
		for i in range(batch_num):
			batch = self.iris_frame.loc[start:start+self.batch_size-1,:]
			start = start + self.batch_size
			self.batchs.append(batch)


		# for batch in self.batchs[:1]:
		# 		self.input = [np.reshape(i, (4,1)) for i in batch[['sepal_length',
		# 															'sepal_width',
		# 															'petal_length',
		# 															'petal_width']].values]

		# 		self.input = self.input / self.max_for_normalize

		# 		self.output = [self.vectorized(o) for o in batch['species'].values]



		# 		print('input.shape', self.input[0].shape)
		# 		print('input', self.input[0])
		# 		print('\n')
		# 		print('output.shape', self.output[0].shape)
		# 		print('output', self.output[0])
		# 		print('\n')
		# 		print('data length', len(self.input))


	def vectorized(self, j):
		e = np.zeros((3,1))
		e[j] = 1
		return e


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


	def error(self, input, output):
		'''
			use mean square error
		'''
		self.feed_forward(input)
		self.e = np.square( self.a_nn[len(self.a_nn)-1]-output )/2
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
		# print('weight_l1_to_l2.shape', weight_l1_to_l2.shape)
		# print('delta_l2.shape', delta_l2.shape)
		# print('a_l1.shape', a_l1.shape)
		delta_l1 = np.dot(weight_l1_to_l2, delta_l2)*a_l1*(1-a_l1)
		return delta_l1


	def delta_network(self, output, log=False):
		
		self.deltas = []

		# calculate delta at final layer
		a_output = self.a_nn[len(self.a_nn)-1]
		delta_output = (a_output - output)*a_output*(1-a_output)
		self.deltas.append(delta_output)

		# calculate delta of hiden layer
		delta_l2 = delta_output
		for i in range(len(self.a_nn)-1):
			idx = len(self.a_nn) - 1 - i
			
			a_l1 = self.a_nn[idx-1]
			weight_l1_to_l2 = self.weights[idx]
			delta_l2 = self.delta_layer(a_l1, weight_l1_to_l2, delta_l2)

			self.deltas.append(delta_l2)

		self.deltas.reverse()

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
		d_weight = input * self.deltas[0].transpose()
		d_bias = self.deltas[0]
		self.d_weights.append(d_weight)
		self.d_biases.append(d_bias)

		if log:
			print('calculate derivative')
			print('\n')
			print('self.a_nn')
			for a in self.a_nn:
				print(a.shape)

			print('self.deltas')
			for d in self.deltas:
				print(d.shape)

		for i in range(len(self.a_nn)-1):
			if log:
				print('self.deltas[len(self.a_nn)-1-i].shape', self.deltas[len(self.a_nn)-1-i].shape)
				print('self.a_nn[i].transpose().shape', self.a_nn[i].transpose().shape)

			d_weight = self.a_nn[i] * self.deltas[i+1].transpose()
			d_bias = self.deltas[i+1]

			self.d_weights.append(d_weight)
			self.d_biases.append(d_bias)

		if log:
			print('derivative weights ----------------------')
			for d in self.d_weights:
				print('d_weight.shape', d.transpose().shape)
				# print('d_weight', d.transpose())
				print('\n')

			for d in self.d_biases:
				print('d_bias.shape', d.shape)
				print('d_bias', d)
				print('\n')

	def update_weight_bias(self, log=False):
		# update

		if log:
			print('self.weights')
			for w in self.weights:
				print('w.shape', w.shape)

			print('\n')
			print('self.d_weights')
			for d in self.d_weights:
				print('d.shape', d.shape)

			print('\n')

		for i in range(len(self.weights)-1):
			if log:
				print('self.weights[i].shape', self.weights[i].shape)
				print('self.d_weights[i].shape', self.d_weights[i].shape)
			self.weights[i] = self.weights[i] - self.learning_rate*self.d_weights[i]
			self.biases[i] = self.biases[i] - self.learning_rate*self.d_biases[i]


	def train(self, log=False):
		self.errors = []

		for i in range(self.epoch):
			
			e_epoch = 0
			
			self.shuffle_data_in_batch()

			for batch in self.batchs:
				# for batch in self.batchs[:1]:
		# 		self.input = [np.reshape(i, (4,1)) for i in batch[['sepal_length',
		# 															'sepal_width',
		# 															'petal_length',
		# 															'petal_width']].values]

		# 		self.input = self.input / self.max_for_normalize

		# 		self.output = [self.vectorized(o) for o in batch['species'].values]
		
		# 		print('input.shape', self.input[0].shape)
		# 		print('input', self.input[0])
		# 		print('\n')
		# 		print('output.shape', self.output[0].shape)
		# 		print('output', self.output[0])
		# 		print('\n')
		# 		print('data length', len(self.input))				

			for idx, input in enumerate(self.input):
				output = self.output[idx]
				self.feed_forward(input, log)
				self.delta_network(output, log)
				self.derivative(input, log)
				self.update_weight_bias(log)
				self.error(input, output)
				e_epoch = e_epoch + self.e_sum
			
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
			print(len(self.a_nn))
			print(self.a_nn[0])
			print(self.a_nn[1])
			print(self.a_nn[2])
			# for i in range(len(self.a_nn)):
			# 	print('a', self.a_nn[i])
			# 	print('a.shape', self.a_nn[i])
			# 	print('\n')

			print('feed forward z -------------------')
			print(len(self.a_nn))
			print(self.z_nn[0])
			print(self.z_nn[1])
			print(self.z_nn[2])
			# for i in self.z_nn:				
			# 	print('z', z)
			# 	print('z.shape', z.shape)
			# 	print('\n')

	def predic(self):
		idxs = [33, 44, 2, 10, 12, 60, 12, 110, 130]

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



# higher at middle is better
# a good model for iris
network = [4,100,3]
learning_rate = 0.3
epoch = 1000
batch_size = 5
load_model = False


nn = NeuralNetwork( network,
					learning_rate,					
					epoch,
					load_model,
					batch_size)

# nn.feed_forward(log=True, input=input)
# nn.delta_network(log=True, output=output)
# nn.derivative(log=True, input=input)
# nn.update_weight_bias()
# nn.error()

# nn.train(log=False)
# nn.save_model()
