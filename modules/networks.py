from chainer import Chain, Sequential
from modules.links import EqualizedLinear, LeakyReluLink
from modules.chains import FirstSynthesisNetwork, SynthesisNetwork

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
			#self.rgb =

	def __call__(self, w)

# Generator network
class Generator(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.m = StyleMapper()
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
