from math import sqrt
from chainer import Chain, Link
from chainer.links import Linear, Convolution2D
from chainer.functions import leaky_relu, pad
from chainer.initializers import Normal

# Learning rate-equalized FC layer
class EqualizedLinear(Chain):

	def __init__(self, in_size, out_size=None, initial_bias=None, gain=sqrt(2)):
		super().__init__()
		self.c = gain * sqrt(1 / in_size)
		with self.init_scope():
			self.linear = Linear(in_size, out_size, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.linear(self.c * x)

# Learning rate-equalized convolution layer
class EqualizedConvolution2D(Chain):

	def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, nobias=False, initial_bias=None, gain=sqrt(2), reflect=False):
		super().__init__()
		self.c = gain * sqrt(1 / (in_channels * ksize ** 2))
		self.pad = pad
		self.reflect = reflect
		with self.init_scope():
			self.conv = Convolution2D(in_channels, out_channels, ksize, stride, 0 if reflect else pad, nobias=nobias, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		if self.reflect and self.pad > 0:
			return self.conv(pad(self.c * x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="symmetric"))
		else:
			return self.conv(self.c * x)

# Leaky ReLU activation function as link
class LeakyReluLink(Link):

	def __init__(self, a):
		super().__init__()
		self.a = a

	def __call__(self, x):
		return leaky_relu(x, self.a)

# Clamped linear interpolation for array
class LerpBlendLink(Link):

	def __init__(self):
		super().__init__()

	def __call__(self, x, y, t):
		if t <= 0: return x
		if t >= 1: return y
		return (1 - t) * x + t * y
