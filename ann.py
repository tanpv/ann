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

class ANN():

	def __init__(self, network_structure, training_data):
		'''
			- network_structure define in detail number of layer and number of neuron at each layer
		'''
		self.network_structure = network_structure
		self.num_layers = len(network_structure)
		self.training_data = training_data
		
	def init_weight_and_bias(self, network_structure):
		self.biases = []
		self.weights = []

		# init biases
		# input layer do not have bias
		# bias is a matrix with 
		# (row, column) = (number_neuron_in_current_layer, 1)
		for num_neuron_in_current_layter in network_structure[1:]:
			bias = np.random.randn(num_neuron_in_current_layter, 1)
			print(bias)
			self.biases.append(bias)

		print('bias')
		print(self.biases)

		# init weights
		# input layer do not have weight
		# weight is a matrix with 
		# (row, column) = (number_neuron_in_current_layer, number_neuron_in_previous_layer)
		for num_neuron_in_current_layter, num_neuron_in_previous_layer in zip(network_structure[1:], network_structure[:-1]):
			weight = np.random.randn(num_neuron_in_current_layter, num_neuron_in_previous_layer)
			print(weight)
			self.weights.append(weight)

		print('weight')
		print(self.weights)


	def train(self):
		
		self.init_weight_and_bias(self.network_structure)
		
		training_data = list(self.training_data)

		for data_sample, idx in zip(training_data,range(len(training_data))):
			input_data = data_sample[0]
			print(input_data)

			output_data = data_sample[1]
			print(output_data)

			self.feed_foward(input_data)
			self.cost(output_data, idx)

			self.back_propagate(output_data)


	def feed_foward(self, input_data):
		'''
			- calculate z at each neuron
			- calculate activation at each neuron
		'''
		self.zs = []
		self.activations = []

		# calculate z at each neuron
		# z = w_current_layer dot a_previous_layer
		# activation = activation_function(z)
		activation = input_data
		for i in range(self.num_layers-1):
			weight = self.weights[i]
			print('weight', weight)
			print('activation', activation)
			print('bias', self.biases[i])
			z = np.dot(weight, activation) + self.biases[i]
			print('z', z)
			self.zs.append(z)
			activation = self.sigmoid(z)
			self.activations.append(activation)


	def cost(self, output_data, idx):
		'''
			- calculate cost at end of feed forward
		'''
		self.cost = 0.5*np.sum((self.activations[-1] - output_data)**2)
		print('\n')
		print('cost after {0} update: {1}'.format(idx, self.cost))


	def back_propagate(self, output_data):
		# init deltas list
		self.deltas = [None]*(self.num_layers-1)
		
		# calculate delta of output layer
		delta = (self.activations[-1] - output_data)*sigmoid_derivative(self.zs[-1])
		self.deltas[-1] = delta
		print(delta)

		# calculate delta of hiden layer
		for idx in range(2, self.num_layers-1):
			

	def cost_derivative_respect_to_weight(self):
		pass

	def cost_derivative_respect_to_bias(self):
		pass

	def update_weight_and_bias(self):
		pass

	def sigmoid(self, z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_derivative(self, z):
		return sigmoid(z)*(1-sigmoid(z))



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

	iris_input = [np.reshape(i, (4,1)) for i in iris_frame[['sepal_length','sepal_width','petal_length','petal_width']].values]
	iris_output = [vectorized_result(o) for o in iris_frame['species'].values]
	
	print(iris_output)
	print(iris_input)

	# first return only one training data point
	return zip(iris_input[:1], iris_output[:1])

def vectorized_result(j):
	e = np.zeros((3,1))
	e[j] = 1
	return e



# ---------------------------------------------

training_data = load_iris_data()

# 4 neuron input
# 3 neuron output
# 5 neuron hiden layer
network_structure = [4,2,3]

n = ANN(network_structure, training_data)
n.train()

# ---------------------------------------------

















