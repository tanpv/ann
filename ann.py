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

	def __init__(self, size):
		'''
			- size is a list define ann structure
			- init the weight and bias
		'''
		self.num_layers = len(size)
		


	def feed_foward(self):
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


print(list(load_iris_data()))

# 4 neuron input
# 3 neuron output
# 5 neuron hiden layer
size = [4,3,3]
# n = ANN(size)
















