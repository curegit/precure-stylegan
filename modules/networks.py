import numpy as np
from chainer import Chain, Sequential
from chainer.functions import mean, sqrt
from modules.links import EqualizedLinear, LeakyReluLink
from modules.gchains import InitialSynthesisNetwork, SynthesisNetwork
from modules.dchains import DiscriminatorChain, FinalDiscriminatorChain, Downsampler

# Feature mapping network
class FeatureMapper(Chain):

	def __init__(self, size, depth):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyReluLink(0.2)).repeat(depth)

	def __call__(self, z):
		return self.mlp(self.normalize(z))

	def normalize(self, z):
		return z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-8)

# Image generation network
class ImageGenerator(Chain):

	def __init__(self, w_size, init_channel=512, final_channel=16, max_stage=9):
		super().__init__()
		with self.init_scope():
			if max_stage >= 1: self.s1 = InitialSynthesisNetwork(init_channel, min(init_channel, final_channel * 2 ** (max_stage - 1)), w_size, 4, 4)
			if max_stage >= 2: self.s2 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 1)), min(init_channel, final_channel * 2 ** (max_stage - 2)), w_size)
			if max_stage >= 3: self.s3 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 2)), min(init_channel, final_channel * 2 ** (max_stage - 3)), w_size)
			if max_stage >= 4: self.s4 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 3)), min(init_channel, final_channel * 2 ** (max_stage - 4)), w_size)
			if max_stage >= 5: self.s5 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 4)), min(init_channel, final_channel * 2 ** (max_stage - 5)), w_size)
			if max_stage >= 6: self.s6 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 5)), min(init_channel, final_channel * 2 ** (max_stage - 6)), w_size)
			if max_stage >= 7: self.s7 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 6)), min(init_channel, final_channel * 2 ** (max_stage - 7)), w_size)
			if max_stage >= 8: self.s8 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 7)), min(init_channel, final_channel * 2 ** (max_stage - 8)), w_size)
			if max_stage >= 9: self.s9 = SynthesisNetwork(min(init_channel, final_channel * 2 ** (max_stage - 8)), final_channel, w_size)

	def __call__(self, w, stage, alpha=1.0):
		blend = 0 <= alpha < 1
		last, up = stage == 1, stage == 2 and blend
		h1, rgb = self.s1(w, last, up)
		if last: return h1
		last, up = stage == 2, stage == 3 and blend
		h2, rgb = self.s2(h1, w, last, up, alpha, rgb)
		if last: return h2
		last, up = stage == 3, stage == 4 and blend
		h3, rgb = self.s3(h2, w, last, up, alpha, rgb)
		if last: return h3
		last, up = stage == 4, stage == 5 and blend
		h4, rgb = self.s4(h3, w, last, up, alpha, rgb)
		if last: return h4
		last, up = stage == 5, stage == 6 and blend
		h5, rgb = self.s5(h4, w, last, up, alpha, rgb)
		if last: return h5
		last, up = stage == 6, stage == 7 and blend
		h6, rgb = self.s6(h5, w, last, up, alpha, rgb)
		if last: return h6
		last, up = stage == 7, stage == 8 and blend
		h7, rgb = self.s7(h6, w, last, up, alpha, rgb)
		if last: return h7
		last, up = stage == 8, stage == 9 and blend
		h8, rgb = self.s8(h7, w, last, up, alpha, rgb)
		if last: return h8
		last, up = stage == 9, stage >= 10
		h9, rgb = self.s9(h8, w, last, up, alpha, rgb)
		if last: return h9
		return rgb

# Generator network
class Generator(Chain):

	def __init__(self, z_size=512, depth=8, channels=(512, 16), max_stage=9):
		super().__init__()
		self.z_size = z_size
		with self.init_scope():
			self.mapper = FeatureMapper(z_size, depth)
			self.generator = ImageGenerator(z_size, *channels, max_stage)

	def __call__(self, z, stage, alpha=1.0):
		return self.generator(self.mapper(z), stage, alpha)

	def generate_latent(self, batch):
		return self.xp.random.normal(size=(batch, self.z_size)).astype(self.xp.float32) if self.xp == np else self.xp.random.normal(size=(batch, self.z_size), dtype=self.xp.float32)

	def resolution(self, stage):
		return (2 * 2 ** stage, 2 * 2 ** stage)

# Discriminator network
class Discriminator(Chain):

	def __init__(self, channels=(512, 16), max_stage=9):
		super().__init__()
		with self.init_scope():
			self.ds = Downsampler()
			final_channel, init_channel = channels
			if max_stage >= 9: self.d1 = DiscriminatorChain(init_channel, min(init_channel * 2 ** (max_stage - 8), final_channel))
			if max_stage >= 8: self.d2 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 8), final_channel), min(init_channel * 2 ** (max_stage - 7), final_channel))
			if max_stage >= 7: self.d3 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 7), final_channel), min(init_channel * 2 ** (max_stage - 6), final_channel))
			if max_stage >= 6: self.d4 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 6), final_channel), min(init_channel * 2 ** (max_stage - 5), final_channel))
			if max_stage >= 5: self.d5 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 5), final_channel), min(init_channel * 2 ** (max_stage - 4), final_channel))
			if max_stage >= 4: self.d6 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 4), final_channel), min(init_channel * 2 ** (max_stage - 3), final_channel))
			if max_stage >= 3: self.d7 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 3), final_channel), min(init_channel * 2 ** (max_stage - 2), final_channel))
			if max_stage >= 2: self.d8 = DiscriminatorChain(min(init_channel * 2 ** (max_stage - 2), final_channel), final_channel)
			if max_stage >= 1: self.d9 = FinalDiscriminatorChain(final_channel)

	def __call__(self, x, stage, alpha=1.0):
		blend = 0 <= alpha < 1
		h1 = self.d1(x, stage == 9) if stage >= 9 else x
		h2 = self.d2(h1, stage == 8, alpha, self.ds(x) if stage == 9 and blend else None) if stage >= 8 else h1
		h3 = self.d3(h2, stage == 7, alpha, self.ds(x) if stage == 8 and blend else None) if stage >= 7 else h2
		h4 = self.d4(h3, stage == 6, alpha, self.ds(x) if stage == 7 and blend else None) if stage >= 6 else h3
		h5 = self.d5(h4, stage == 5, alpha, self.ds(x) if stage == 6 and blend else None) if stage >= 5 else h4
		h6 = self.d6(h5, stage == 4, alpha, self.ds(x) if stage == 5 and blend else None) if stage >= 4 else h5
		h7 = self.d7(h6, stage == 3, alpha, self.ds(x) if stage == 4 and blend else None) if stage >= 3 else h6
		h8 = self.d8(h7, stage == 2, alpha, self.ds(x) if stage == 3 and blend else None) if stage >= 2 else h7
		return self.d9(h8, stage == 1, alpha, self.ds(x) if stage == 2 and blend else None) if stage >= 1 else h8
