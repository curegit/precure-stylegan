from chainer import Chain, Sequential
from chainer.functions import mean, sqrt
from modules.links import EqualizedLinear, LeakyReluLink
from modules.gchains import InitialSynthesisNetwork, SynthesisNetwork
from modules.dchains import DiscriminatorChain, FinalDiscriminatorChain
from modules.functions import normal_random

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
			self.s1 = InitialSynthesisNetwork(512, 512, w_size, 4, 4)
			self.s2 = SynthesisNetwork(512, 512, w_size)
			self.s3 = SynthesisNetwork(512, 512, w_size)
			self.s4 = SynthesisNetwork(512, 512, w_size)
			self.s5 = SynthesisNetwork(512, 256, w_size)
			self.s6 = SynthesisNetwork(256, 128, w_size)
			self.s7 = SynthesisNetwork(128, 64, w_size)
			self.s8 = SynthesisNetwork(64, 32, w_size)
			self.s9 = SynthesisNetwork(32, 16, w_size)

	def __call__(self, w, stage):
		last = stage == 1
		h1 = self.s1(w, last)
		if last: return h1
		last = stage == 2
		h2 = self.s2(h1, w, last)
		if last: return h2
		last = stage == 3
		h3 = self.s3(h2, w, last)
		if last: return h3
		last = stage == 4
		h4 = self.s4(h3, w, last)
		if last: return h4
		last = stage == 5
		h5 = self.s5(h4, w, last)
		if last: return h5
		last = stage == 6
		h6 = self.s6(h5, w, last)
		if last: return h6
		last = stage == 7
		h7 = self.s7(h6, w, last)
		if last: return h7
		last = stage == 8
		h8 = self.s8(h7, w, last)
		if last: return h8
		last = stage == 9
		h9 = self.s9(h8, w, last)
		if last: return h9

# Generator network
class Generator(Chain):

	def __init__(self, z_size):
		super().__init__()
		self.z_size = z_size
		with self.init_scope():
			self.m = FeatureMapper(z_size, 8)
			self.g = ImageGenerator(z_size)

	def __call__(self, z, stage):
		return self.g(self.m(z), stage)

	def generate_latent(self, batch):
		return normal_random(shape=(batch, self.z_size))

# Discriminator network
class Discriminator(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.d1 = DiscriminatorChain(16, 32)
			self.d2 = DiscriminatorChain(32, 64)
			self.d3 = DiscriminatorChain(64, 128)
			self.d4 = DiscriminatorChain(128, 256)
			self.d5 = DiscriminatorChain(256, 512)
			self.d6 = DiscriminatorChain(512, 512)
			self.d7 = DiscriminatorChain(512, 512)
			self.d8 = DiscriminatorChain(512, 512)
			self.d9 = FinalDiscriminatorChain(512)

	def __call__(self, x, stage):
		h1 = self.d1(x, stage == 9) if stage >= 9 else x
		h2 = self.d2(h1, stage == 8) if stage >= 8 else h1
		h3 = self.d3(h2, stage == 7) if stage >= 7 else h2
		h4 = self.d4(h3, stage == 6) if stage >= 6 else h3
		h5 = self.d5(h4, stage == 5) if stage >= 5 else h4
		h6 = self.d6(h5, stage == 4) if stage >= 4 else h5
		h7 = self.d7(h6, stage == 3) if stage >= 3 else h6
		h8 = self.d8(h7, stage == 2) if stage >= 2 else h7
		return self.d9(h8, stage == 1) if stage >= 1 else h8
