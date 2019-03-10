'''
	- working with iris data set
	- working with mnis data set

	- loading iris data from csv
	- create training data with array of input and array of output
	- init weight and bias
'''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ANN():

	def __init__(self, 
				network_structure, 
				training_data, 
				learning_rate,
				epoc):
		'''
			- network_structure define in detail number of layer and number of neuron at each layer
		'''
		self.network_structure = network_structure
		self.num_layers = len(network_structure)
		self.training_data = training_data
		self.learning_rate = learning_rate
		self.epoc = epoc
		self.cost = []

		
	def init_weight_and_bias(self, network_structure):
		self.biases = [np.random.randn(y, 1) for y in network_structure[1:]]
		print('bias')
		print(self.biases)

		self.weights = [np.random.randn(y, x)
						for x, y in zip(network_structure[:-1], network_structure[1:])]
		print('weight')
		print(self.weights)


	def train(self):		
		self.init_weight_and_bias(self.network_structure)
		
		training_data = list(self.training_data)

		for e in range(self.epoc):
			
		# 	# this is important
			random.shuffle(training_data)

		# for e in range(50000):
			for one_sample, idx in zip(training_data,range(len(training_data))):
				input_sample = one_sample[0]
				# print('input', input_sample)
				output_sample = one_sample[1]
				# print('output', output_sample)

				self.feed_forward(input_sample)
				self.cost_calculate(output_sample, idx)
				self.back_propagate(output_sample)
				# self.cost_derivative_respect_to_bias()
				# self.cost_derivative_respect_to_weight()
				self.update_weight_and_bias()

			self.evaluate_after_each_epoc(training_data)

		self.plot_cost()


	def plot_cost(self):
		plt.figure('cost over time')
		plt.plot(self.cost)
		plt.show()

	def feed_forward_one_sample(self, a):
		for b, w in zip(self.biases, self.weights):
			a = self.sigmoid(np.dot(w, a)+b)
		return a

	def feed_forward(self, input_sample):
		'''
			- calculate z at each neuron
			- calculate activation at each neuron
		'''
		self.zs = []
		self.activations = []

		# calculate z at each neuron
		# z = w_current_layer dot a_previous_layer
		# activation = activation_function(z)
		activation = input_sample
		self.activations.append(activation)
		for i in range(self.num_layers-1):
			weight = self.weights[i]
			# print('weight', weight)
			# print('activation', activation)
			# print('bias', self.biases[i])
			z = np.dot(weight, activation) + self.biases[i]
			# print('z', z)
			self.zs.append(z)
			activation = self.sigmoid(z)
			self.activations.append(activation)

		# print('z', self.zs)
		# print('activation', self.activations)


	def cost_calculate(self, output_data, idx):
		'''
			- calculate cost at end of feed forward
		'''
		# print(self.activations[-1])
		# print(output_data)
		# self.activations[-1] - output_data
		cost_value = 0.5*np.sum((output_data - self.activations[-1])**2)
		# print('\n')
		print('cost after {0} update: {1}'.format(idx, cost_value))
		self.cost.append(cost_value)


	def back_propagate(self, output_data):
		# 
		self.cost_derivative_respect_to_weights = [np.zeros(b.shape) for b in self.biases]
		self.cost_derivative_respect_to_biases = [np.zeros(w.shape) for w in self.weights]

		# final layer
		delta = (self.activations[-1] - output_data)*self.sigmoid_derivative(self.zs[-1])
		# print('delta', delta)
		# print('activation', self.activations[-2].transpose())
		cost_derivative_respect_to_weight = np.dot(delta, self.activations[-2].transpose())
		self.cost_derivative_respect_to_weights[-1] = cost_derivative_respect_to_weight
		self.cost_derivative_respect_to_biases[-1] = delta

		# calculate delta of hiden layer
		for idx in range(2, self.num_layers):
			# print(delta)
			delta = np.dot(self.weights[-idx+1].transpose(),delta)*self.sigmoid_derivative(self.zs[-idx])
			cost_derivative_respect_to_weight = np.dot(delta, self.activations[-idx-1].transpose())
			self.cost_derivative_respect_to_weights[-idx] = cost_derivative_respect_to_weight
			self.cost_derivative_respect_to_biases[-idx] = delta
		

	def cost_derivative_respect_to_bias(self):
		self.cost_derivative_respect_to_biases = self.deltas


	def cost_derivative_respect_to_weight(self):
		self.cost_derivative_respect_to_weights = [None]*(self.num_layers-1)

		# wrong here
		for idx in range(0, self.num_layers-1):
			# print(len(self.deltas))
			# print(len(self.activations))
			# print(idx)
			devivative = np.dot(self.deltas[idx], self.activations[idx].transpose())
			self.cost_derivative_respect_to_weights[idx]=devivative


	def update_weight_and_bias(self):
		# print('update weight and bias')
		for idx in range(len(self.weights)):
			self.weights[idx] = self.weights[idx] - self.learning_rate*self.cost_derivative_respect_to_weights[idx]
			self.biases[idx] = self.biases[idx] - self.learning_rate*self.cost_derivative_respect_to_biases[idx]


	def evaluate_after_each_epoc(self, training_data):
		true_number = 0
		total = len(training_data)
		# total = 100
		for x,y in training_data[:]:
			activation = self.feed_forward_one_sample(x)
			idx_activation = np.argmax(activation)
			# print('activation', activation)
			# print('idx_activation', idx_activation)
			idx_y = np.argmax(y)
			# print('y', y)
			# print('idx_y', idx_y)
			if idx_activation == idx_y:
				true_number = true_number + 1
				# print('true prediction')
			# else:
				# print('false prediction')

		print('total true prediction {0} / {1}'.format(true_number, total))

	def sigmoid(self, z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_derivative(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

def load_iris_data():
	'''
		- load data from csv file
		- build input array and output array
	'''
	iris_frame = pd.read_csv('iris.csv')
	
	# convert species from text to number
	iris_frame.loc[iris_frame['species']=='setosa', 'species'] = 0
	iris_frame.loc[iris_frame['species']=='versicolor', 'species'] = 1
	iris_frame.loc[iris_frame['species']=='virginica', 'species'] = 2

	max_for_normalize = iris_frame[['sepal_length',
									'sepal_width',
									'petal_length',
									'petal_width']].values.max()

	# print('max for normalize', max_for_normalize)

	iris_input = [np.reshape(i, (4,1)) / max_for_normalize for i in iris_frame[['sepal_length',
															'sepal_width',
															'petal_length',
															'petal_width']].values]

	iris_output = [vectorized_iris_result(o) for o in iris_frame['species'].values]
	
	print('output', iris_output[:1])
	print('input', iris_input[:1])

	return zip(iris_input, iris_output)

def vectorized_mnist_result(j):
	e = np.zeros((10,1))
	e[j] = 1
	return e

def vectorized_iris_result(j):
	e = np.zeros((3,1))
	e[j] = 1
	return e

def load_mnist_data():
	mnist_frame = pd.read_csv('mnist_train.csv')
	mnist_input = [np.reshape(i, (784,1))/255 for i in mnist_frame.iloc[:,1:].values]
	print(mnist_input[:1])
	mnist_output = [vectorized_mnist_result(o) for o in mnist_frame.iloc[:,0].values]
	return zip(mnist_input, mnist_output)

# ---------------------------------------------
# # mnist data
# training_data = load_mnist_data()
# network_structure = [784,30,10]
# learning_rate = 3.0
# epoc = 30

# iris data
training_data = load_iris_data()
network_structure = [4,6,3]
learning_rate = 1
epoc = 30

n = ANN(network_structure, training_data, learning_rate, epoc)
n.train()
# ---------------------------------------------

















