# --------------------------------------------------
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
		self.init_bias()
	
	def init_weight(self):
		self.w = np.random.randn()

	def init_bias(self):
		self.b = np.random.randn()
	
	def feed_forward(self):
		self.output = self.input*self.w + self.b

	def error(self):
		# use mean square error
		return np.square( self.output - self.target)/2
	
	def error_derivative_respect_to_weight(self):
		self.d_w = self.input*(self.output - self.target)

	def error_derivative_respect_to_bias(self):
		self.d_b = self.output - self.target

	def train(self):
		self.errors = []
		self.ws = []
		self.d_ws = []
		for n in range(self.epoch):

			# training process			
			self.feed_forward()
			
			self.error_derivative_respect_to_weight()
			self.error_derivative_respect_to_bias()

			self.w = self.w - self.learning_rate*self.d_w
			self.b = self.b - self.learning_rate*self.d_b
			self.ws.append(self.w)
			self.d_ws.append(self.d_w)

			# calculate error after update weight
			error = self.error()
			self.errors.append(error)
			print('error at epoch {0}'.format(n), error)
			print('w', self.w)
			print('b', self.b)
			print('\n')

		self.plot_train_error()

	
	def plot_train_error(self):		
		plt.xticks(np.arange(0, self.epoch, 1))
		plt.plot(range(self.epoch), self.errors)
		# plt.plot(range(self.epoch), self.ws	)
		# plt.plot(range(self.epoch), self.d_ws)
		plt.show()

# ---------------------------------
# ---------------------------------
# ---------------------------------

input = [1]
target = [2]
learning_rate = 0.1
epoch = 50

n = Neuron(	input,
			target,
			learning_rate,
			epoch )

n.train()

# ---------------------------------
# ---------------------------------
# ---------------------------------
