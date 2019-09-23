# import numpy as np
# import matplotlib.pyplot as plt

# class Neuron(object):
# 	def __init__(self, 
# 				input,
# 				output,
# 				learning_rate,
# 				epoch):

# 		self.input = input
# 		self.output = output
# 		self.learning_rate = learning_rate
# 		self.epoch = epoch

# 		self.init_weight()
# 		self.init_bias()

# 	def init_weight(self):
# 		self.weight = np.random.randn()
# 		print('weight', self.weight)

# 	def init_bias(self):
# 		self.bias = np.random.randn()
# 		print('bias', self.bias)

# 	def feed_forward(self, input):
# 		return self.sigmoid(input*self.weight + self.bias)

# 	def z(self, input, weight, bias):
# 		# z = input * weight + bias
# 		z = input*weight + bias
# 		return z

# 	def sigmoid(self, z):
# 		return 1 / (1 + np.exp(-z))

# 	def sigmoid_derivative_respect_to_z(self, z):
# 		return z*(1-z)

# 	def loss(self):
# 		# caculate loss with mean square error
# 		# mse = square(expected_output - real_output)
# 		return np.square( self.feed_forward(self.input[0]) - self.output[0])

# 	def loss_derivative_respect_to_weight(self, weight, bias):
# 		z = self.input[0] * weight + bias
# 		return 2*(self.sigmoid(z)-self.output[0])*self.sigmoid(z)*(1-self.sigmoid(z))*self.input[0]

# 	def loss_derivative_respect_to_bias(self, weight, bias):
# 		z = self.input[0] * weight + bias
# 		return 2*(self.sigmoid(z)-self.output[0])*self.sigmoid(z)*(1-self.sigmoid(z))

# 	def train(self):
# 		self.losses = []
# 		for n in range(self.epoch):

# 			# calculate current loss
# 			loss = self.loss()
# 			print('error at epoch {0} is {1}'.format(n, loss))
# 			self.losses.append(loss)

# 			# update weight and bias
# 			self.weight = self.weight - self.learning_rate*self.loss_derivative_respect_to_weight(self.weight, self.bias)
# 			self.bias = self.bias - self.learning_rate*self.loss_derivative_respect_to_bias(self.weight, self.bias)

# 			print('updated weight {0}'.format(self.weight))
# 			print('updated bias {0}'.format(self.bias))
# 			print('\n')

# 		self.plot_train_error()
  
# 	def plot_train_error(self):		
# 		plt.xticks(np.arange(0, self.epoch, 1))
# 		plt.plot(range(self.epoch), self.losses)
# 		plt.show()


# # note that output of sigmoid function is between (0,1)
# input = [3]
# output = [0.3]
# learning_rate = 0.2
# epoch = 50

# n = Neuron(	input,
# 			output,
# 			learning_rate,
# 			epoch )

# n.train()


# ----------------------------------------------------------------------
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
		self.output = self.sigmoid(self.z(self.input, self.w, self.b))

	def z(self, input, weight, bias):
		return input*weight + bias

	def sigmoid_derivative(self, z):
		s = 1/(1+np.exp(-z))
		return s*(1-s)

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def error(self):
		# use mean square error
		return np.square( self.output - self.target)/2
	
	def error_derivative_respect_to_weight(self):
		sigmoid_derivative = self.sigmoid_derivative(self.z(self.input, self.w, self.b))
		self.d_w = self.input*(self.output - self.target)*sigmoid_derivative

	def error_derivative_respect_to_bias(self):
		sigmoid_derivative = self.sigmoid_derivative(self.z(self.input, self.w, self.b))
		self.d_b = (self.output - self.target)*sigmoid_derivative

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

input = [0.2]
target = [0.4]
learning_rate = 0.3
epoch = 200

n = Neuron(	input,
			target,
			learning_rate,
			epoch )

n.train()

# ---------------------------------
# ---------------------------------
# ---------------------------------



