from chainer import Chain, Sequential
from chainer.functions import mean, sqrt
from modules.links import EqualizedLinear, LeakyReluLink
from modules.chains import FirstSynthesisNetwork, SynthesisNetwork

# Feature mapping network
class FeatureMapper(Chain):

	def __init__(self, size, depth):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyReluLink(0.1)).repeat(depth)

	def __call__(self, z):
		return self.mlp(self.normalize(z))

	def normalize(self, z):
		return z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-8)

# Image generation network
class ImageGenerator(Chain):

	def __init__(self, w_size):
		super().__init__()
		with self.init_scope():
			self.s1 = FirstSynthesisNetwork()
			self.s2 = SynthesisNetwork()
			self.s3 = SynthesisNetwork()
			self.s4 = SynthesisNetwork()
			self.s5 = SynthesisNetwork()
			self.s6 = SynthesisNetwork()
			self.s7 = SynthesisNetwork()
			self.s8 = SynthesisNetwork()
			self.s9 = SynthesisNetwork()


	def __call__(self, w, stage):

		if stage == 1: return



# Generator network
class Generator(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.m = FeatureMapper()
			self.g = ImageGenerator()

	def __call__(self, z):
		return self.g(self.m(z))

	def generate_latent(self, batch):
		# np?
		return self.xp.random.normal(size=(batch, self.size)).astype(np.float32)

# Discriminator network
class Discriminator(Chain):

	def __init__(self):
		super().__init__()
