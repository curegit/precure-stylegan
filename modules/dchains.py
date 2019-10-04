from chainer import Chain
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
class Upsampler(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		return average_pooling_2d(x, ksize=2, stride=2, pad=0)
