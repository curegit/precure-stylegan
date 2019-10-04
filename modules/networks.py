from chainer import Chain, Sequential
from chainer.functions import mean, sqrt
from modules.links import EqualizedLinear, LeakyReluLink
from modules.chains import InitialSynthesisNetwork, SynthesisNetwork
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
