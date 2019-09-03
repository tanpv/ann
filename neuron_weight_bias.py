'''
	- multiple data samples
	- neuron
		- weight
		- bias
		- activation function
	- feed forward
	- mean square error
		- plot 3d for error
		- https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725
	- training
	- learning rate
		- too big with value 1
		- too small with value 0.01
		- normal learning rate at 0.1
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

		self.input0 = input[0]
		self.output0 = output[0]
		self.learning_rate = learning_rate
		self.epoch = epoch

		self.init_weight()
		self.init_bias()

	def init_weight(self):
		self.w = np.random.randn()
		print('w', self.w)

	def init_bias(self):
		self.b = np.random.randn()
		print('b', self.b)

	def feed_forward(self):
		return self.input0*self.w+self.b

	def mean_square_error(self):
		return np.square( self.feed_forward() - self.output0)

	def gradient_w(self, w):
		return 2*self.input0*(w*self.input0 - self.output0)

	def gradient_b(self, b):
		return 2*(b + self.input0*self.w - self.output0)

	def train(self):
		self.errors = []
		for n in range(self.epoch):
			print('error at epoch {0}'.format(n), self.mean_square_error())

			self.errors.append(self.mean_square_error())
			self.w = self.w - self.learning_rate*self.gradient_w(self.w)
			self.b = self.b - self.learning_rate*self.gradient_b(self.b)

			print('updated weight {0}'.format(self.w))
			print('updated bias {0}'.format(self.b))
			print('\n')

		self.plot_train_error()

	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


input = [1]
output = [2]

learning_rate = 0.01

learning_rate = 0.1

epoch = 50

n = Neuron(	input,
			output,
			learning_rate,
			epoch )
n.train()
