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

	def __init__(self, network_structure):
		'''
			- network_structure define in detail number of layer and number of neuron at each layer
		'''
		self.init_weight_and_bias(network_structure)
		

	def init_weight_and_bias(self, network_structure):
		self.num_layers = len(network_structure)
		self.biases = []
		self.weights = []

		# init biases
		# input layer do not have bias
		# bias is a matrix with 
		# (row, column) = (number_neuron_in_layer, 1)
		for num_neuron in network_structure[1:]:
			bias = np.random.randn(num_neuron, 1)
			print(bias)
			self.biases.append(bias)

		print('bias')
		print(self.biases)

		# init weights
		# input layer do not have weight
		# weight is a matrix with 
		# (row, column) = (number_neuron_in_current_layer, number_neuron_in_previous_layer)
		for num_neuron, num_neuron_previous in zip(network_structure[1:], network_structure[:-1]):
			weight = np.random.randn(num_neuron, num_neuron_previous)
			print(weight)
			self.weights.append(weight)

		print('weight')
		print(self.weights)


	def feed_foward(self, network_structure):
		'''
			- calculate z at each neuron
			- calculate activation at each neuron
		'''
		self.zs = []
		self.activations = []

		# calculate z at each neuron
		# z = w dot a
		# activation = activation_function(z)


	def cost_calculate(self):
		'''
			- calculate cost at end of feed forward
		'''
		pass

	def delta_calculate(self):
		pass

	def back_propagate_delta(self):
		pass

	def cost_derivative_respect_to_weight(self):
		pass

	def cost_derivative_respect_to_bias(self):
		pass

	def update_weight_and_bias(self):
		pass

	def sigmoid(self, input):
		pass

	def sigmoid_derivative(self, input):
		pass



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


# print(list(load_iris_data()))

# 4 neuron input
# 3 neuron output
# 5 neuron hiden layer
network_structure = [4,6,3]
n = ANN(network_structure)

















