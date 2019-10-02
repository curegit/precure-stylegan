from chainer import Chain, Sequential
from modules.links import EqualizedLinear, EqualizedConvolution2D, LeakyReluLink

# Mapping network, latent Z to W
class StyleMapper(Chain):

	def __init__(self, size=256, depth=8):
		super().__init__()
		self.size = size
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyReluLink(0.1)).repeat(depth)

	def __call__(self, z):
		# np?
		n = z / self.xp.sqrt(self.xp.sum(z * z, axis=1, keepdims=True) / 8 + 1e-8)
		return self.mlp(n)

	def generate_latent(self, batch):
		return self.xp.random.normal(size=(batch, self.size)).astype(np.float32)

#
class Generator(Chain):

	def __init__():
		nil

#
class Discriminator(Chain):

	def __init__():
		nil
