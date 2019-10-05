from chainer import Chain, Sequential
from chainer.functions import mean, sqrt, concat, broadcast_to, average_pooling_2d
from modules.links import EqualizedLinear, EqualizedConvolution2D, LeakyReluLink

# Link inserting a new channel of mini-batch standard deviation
class MiniBatchStandardDeviation(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		m = broadcast_to(mean(x, axis=0, keepdims=True), x.shape)
		# TODO: why is eps needed?
		sd = sqrt(mean((x - m) ** 2, axis=0, keepdims=True) + 1e-8)
		channel = broadcast_to(mean(sd), (x.shape[0], 1, x.shape[2], x.shape[3]))
		return concat((x, channel), axis=1)

# 1/2 Downsample operation as link
class Downsampler(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		return average_pooling_2d(x, ksize=2, stride=2, pad=0)

# Head blocks of discriminator
class DiscriminatorChain(Chain):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.rgb = Sequential(EqualizedConvolution2D(3, in_channels, ksize=1, stride=1, pad=0), LeakyReluLink(0.2))
			self.c1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
			self.r1 = LeakyReluLink(0.2)
			self.c2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.r2 = LeakyReluLink(0.2)
			self.d1 = Downsampler()

	def __call__(self, x, first):
		h1 = self.rgb(x) if first else x
		h2 = self.c1(h1)
		h3 = self.r1(h2)
		h4 = self.c2(h3)
		h5 = self.r2(h4)
		return self.d1(h5)

# Last block of discriminator
class FinalDiscriminatorChain(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.rgb = Sequential(EqualizedConvolution2D(3, in_channels, ksize=1, stride=1, pad=0), LeakyReluLink(0.2))
			self.mb = MiniBatchStandardDeviation()
			self.c1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
			self.r1 = LeakyReluLink(0.2)
			self.c2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
			self.r2 = LeakyReluLink(0.2)
			self.fc = EqualizedLinear(in_channels, 1)

	def __call__(self, x, first=False):
		h1 = self.rgb(x) if first else x
		h2 = self.mb(h1)
		h3 = self.c1(h2)
		h4 = self.r1(h3)
		h5 = self.c2(h4)
		h6 = self.r2(h5)
		return self.fc(h6)
