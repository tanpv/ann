'''
	- intro
	- weight
	- feed forward
		- find output correspond to input at current value of weight
	- mean square error
	- gradient decense
	- error derivative respect to weight
	- back propagation
	- training
	- learning rate
		- too big with value 1
		- too small with value 0.01
		- normal learning rate at 0.1
	- error and show up error
	- epoch
'''

import numpy as np
import matplotlib.pyplot as plt

class Neuron(object):
	def __init__(self, 
				input,
				output,
				learning_rate,
				epoch):

		self.input = input[0]
		self.target = target[0]
		self.learning_rate = learning_rate
		self.epoch = epoch

		self.init_weight()

	
	def init_weight(self):
		self.w = np.random.randn()
		print(self.w)

	
	def feed_forward(self):
		self.output = self.input*self.w

	def mean_square_error(self):
		return np.square( self.output - self.target)/2

	
	def error_derivative_respect_to_weight(self, w):
		return self.input*(self.output - self.target)

	
	def train(self):
		self.errors = []
		for n in range(self.epoch):

			self.feed_forward()

			self.w = self.w - self.learning_rate*self.error_derivative_respect_to_weight(self.w)

			print('error at epoch {0}'.format(n), self.mean_square_error())

			self.errors.append(self.mean_square_error())
			
			print('updated weight {0}'.format(self.w))

			print('\n')

		self.plot_train_error()

	
	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


# ---------------------------------
input = [1]
target = [2]
learning_rate = 0.1
epoch = 100

n = Neuron(	input,
			target,
			learning_rate,
			epoch )

n.train()
# ---------------------------------



		