'''
	- data sample
	- weight
	- feed forward
		- find output correspond to input at current value of weight
	- mean square error
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
				train_data,
				learning_rate,
				epoch):
		self.input0 = train_data[0][0]
		self.output0 = train_data[0][1]
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.init_weight()

	def init_weight(self):
		self.w = np.random.randn()
		print(self.w)

	def feed_forward(self):
		return self.input0*self.w

	def mean_square_error(self):
		return np.square( self.feed_forward() - self.output0)

	def gradient(self, w):
		return 2*self.input0*(w*self.input0 - self.output0)

	def train(self):
		self.errors = []
		for n in range(self.epoch):
			print('error at epoch {0}'.format(n), self.mean_square_error())
			self.errors.append(self.mean_square_error())
			self.w = self.w - self.learning_rate*self.gradient(self.w)
			print('updated weight {0}', self.w)
			print('\n')

		self.plot_train_error()

	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		plt.show()


train_data = [(1,2)]

learning_rate = 0.01

learning_rate = 0.1

epoch = 20

n = Neuron(	train_data,
			learning_rate,
			epoch )
n.train()






		