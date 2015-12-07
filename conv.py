#############################################################
#            Created by Yinan Xu, 11/29/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import numpy
import theano
from scipy import signal
from scipy import ndimage

class Convolutional(object):

	def __init__(self,input_x,n_size,n_filter):
		self.kernel = numpy.random.randn(n_size, n_size,n_filter)
		self.input = input_x

	def conv_2d(self):
		return ndimage.convolve(self.input, self.kernel, mode='valid')


def test():
	x = numpy.random.rand(5,5,1)
	# print x
	# kernel = numpy.random.rand(3,3)
	# pred = signal.convolve2d(x, kernel, mode='valid')
	# print pred
	conv_new = Convolutional(input_x=x, n_size=3,n_filter=1)
	# print conv_new.kernel
	print conv_new.conv_2d()











if __name__ == '__main__':
    test()
