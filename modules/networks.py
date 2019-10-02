from chainer import Chain, Sequential
from modules.links import EqualizedLinear, LeakyReluLink

# Mapping network
class StyleMapper(Chain):

	def __init__(self, size=256, depth=6):
		super().__init__()
		self.size = size
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyReluLink(0.1)).repeat(depth)

	def __call__(self, z):
		# np? z:(batch, size)
		n = z / self.xp.sqrt(self.xp.sum(z * z, axis=1, keepdims=True) / self.size + 1e-8)
		return self.mlp(n) # (batch, size)

# Image generation network
class ImageGenerator(Chain):

	def __init__(self):
		super().__init__()

# Generator network
class Generator(Chain):

	def __init__(self):
		super().__init__()

	def generate_latent(self, batch):
		# np?
		return self.xp.random.normal(size=(batch, self.size)).astype(np.float32)

# Discriminator network
class Discriminator(Chain):

	def __init__(self):
		super().__init__()
