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
			size is a list define ann structure
		'''
		print(size)
		pass

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



class DataLoader():
	'''
		- load data from csv file
		- build input array and output array
	'''
	def __init__(self, data_file):
		iris = np.loadtxt(data_file, delimiter=",", skiprows=1)
		# second = csv[:,1]
		# third = csv[:,2]
		print(iris)



data_file = 'iris.csv'
dl = DataLoader(data_file)


# size = [2,3,1]
# n = ANN(size)
















