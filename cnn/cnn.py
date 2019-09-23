import numpy as np
import mnist

class Conv3x3(object):
	def __init__(self, num_filters):
		self.num_filters = num_filters

		# init filter value
		# filter is inited difference value at beginner
		self.filters = np.random.randn(num_filters, 3, 3) / 9

		print('self.filters.shape')
		print(self.filters.shape)
		print('self.filters')
		print(self.filters)

	def iterate_regions(self, image):
		h, w = image.shape
		for i in range(h-2):
			for j in range(w-2):
				im_region = image[i:(i+3), j:(j+3)]
				yield im_region, i, j


	def forward(self, input):
		self.last_input = input

		h, w = input.shape
		output = np.zeros((h-2, w-2, self.num_filters))
		print('output.shape', output.shape)

		for im_region, i, j in self.iterate_regions(input):
			for idx_filter, filter in enumerate(self.filters):
				output[i,j,idx_filter] = np.sum(im_region*filter)

		return output


train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape)

