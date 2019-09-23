import matplotlib.pyplot as plt
import numpy as np

class GD():
	def __init__(self):
		pass

	def e(self, w):
		# define error function
		return np.square(w)/2 - 2*w + 2

	def e_gradient(self, w):
		return w-2

	def plot_e(self):
		w = np.linspace(-200, 200, 50)
		plt.plot(w, self.e(w))
		plt.plot(w, self.e_gradient(w))
		# plt.plot(self.ws, self.es)

		plt.scatter(self.ws, self.es, s=np.pi*5, c='red')
		plt.show()

	def gradient_descent(self):
		# # example with w = -100
		# w = -100

		# example with w = 100
		w = 200
		
		# too small learning rate
		learning_rate = 0.001

		# too big learning rate
		learning_rate = 2

		# good learning rate
		learning_rate = 0.2
		
		self.es = []
		self.ws = []
		
		for i in range(100):

			w = w - learning_rate*self.e_gradient(w)
			self.ws.append(w)

			e = self.e(w)			
			self.es.append(e)

			print('w', w)			
			print('e', e)
			print('w = {0} - {1}*({2})'.format(w, learning_rate, self.e_gradient(w)))

			print('\n')

		self.plot_e()

gd = GD()
gd.gradient_descent()
