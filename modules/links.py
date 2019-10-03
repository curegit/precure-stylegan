from chainer import Chain, Link
from chainer.links import Linear, Convolution2D
from chainer.functions import leaky_relu
from chainer.initializers import Normal

# Learning rate-equalized FC layer
class EqualizedLinear(Chain):

	def __init__(self, in_size, out_size, initial_bias=None):
		super().__init__()
		# TODO: np?
		self.c = self.xp.sqrt(2 / in_size)
		with self.init_scope():
			self.linear = Linear(in_size, out_size, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.linear(self.c * x)

# Learning rate-equalized convolution layer
class EqualizedConvolution2D(Chain):

	def __init__(self, in_ch, out_ch, ksize, stride, pad, initial_bias=None):
		super().__init__()
		# TODO: np?
		self.c = self.xp.sqrt(2 / (in_ch * ksize ** 2))
		with self.init_scope():
			self.conv = Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.conv(self.c * x)

# Leaky ReLU activation function as link
class LeakyReluLink(Link):

	def __init__(self, a):
		super().__init__()
		self.a = a

	def __call__(self, x):
		return leaky_relu(x, self.a)

#
class MiniBatchStandardDeviation(Link):

	def __call__(self, x):
		m = broadcast_to(mean(x, axis=0, keepdims=True), x.shape)
		v = mean((x - m) ** 2, axis=0, keepdims=True)
		std = mean(sqrt(v + 1e-8), keepdims=True)
		new_channel = broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
		return F.concat((x, new_channel), axis=1)
