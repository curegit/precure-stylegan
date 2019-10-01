from chainer import Chain, Sequential
from modules.links import EqualizedLinear, EqualizedConvolution2D, LeakyReluLayer

# Mapping network, latent Z to W
class StyleMapper(Chain):

	def __init__(self, size=256, depth=8):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyReluLayer(0.1)).repeat(depth)

	def __call__(self, z):
		# TODO: need normalization
		return self.mlp(z)
