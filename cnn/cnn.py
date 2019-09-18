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
			print('im_region.shape')
			print(im_region.shape)
			print('im_region')
			print(im_region)
			# im_region is broadcast out so it have same shape with filters
			# how to understand this sum function ?
			print('im_region * self.filters', (im_region*self.filters).shape)
			# print('im_region * self.filters', (im_region*self.filters).shape)

			# what is this ?
			output[i,j] = np.sum(im_region * self.filters, axis=(1,2))


		return output


train_images = mnist.train_images()
train_labels = mnist.train_labels()


conv = Conv3x3(2)
output = conv.forward(train_images[0])
print(output.shape)


# e = np.array([[[1, 0],
# 		[0, 0]],
#        	[[1, 1],
#         [1, 0]],
#         [[1, 0],
#         [0, 1]]])

# print(e.shape)
# print(e.sum(axis=(1,2)).shape)


