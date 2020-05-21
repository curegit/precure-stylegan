from math import log
from random import randint
from chainer import Variable, Parameter, Chain, Sequential
from chainer.functions import mean, sqrt, broadcast_to, gaussian
from chainer.initializers import Zero, One
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

	def __call__(self, w, stage, alpha=1.0, mix_w=None, mix_stage=None):
		blend = 0 <= alpha < 1
		ws = w + (w[-1:] * (stage - 1))[0:stage] if type(w) is list else [w] * stage
		mix = mix_stage or (10 if mix_w is None or stage < 2 else randint(2, stage))
		last, up = stage == 1, stage == 2 and blend
		h1, rgb = self.s1(ws[0] if mix > 1 else mix_w, last, up)
		if last: return h1
		last, up = stage == 2, stage == 3 and blend
		h2, rgb = self.s2(h1, ws[1] if mix > 2 else mix_w, last, up, alpha, rgb)
		if last: return h2
		last, up = stage == 3, stage == 4 and blend
		h3, rgb = self.s3(h2, ws[2] if mix > 3 else mix_w, last, up, alpha, rgb)
		if last: return h3
		last, up = stage == 4, stage == 5 and blend
		h4, rgb = self.s4(h3, ws[3] if mix > 4 else mix_w, last, up, alpha, rgb)
		if last: return h4
		last, up = stage == 5, stage == 6 and blend
		h5, rgb = self.s5(h4, ws[4] if mix > 5 else mix_w, last, up, alpha, rgb)
		if last: return h5
		last, up = stage == 6, stage == 7 and blend
		h6, rgb = self.s6(h5, ws[5] if mix > 6 else mix_w, last, up, alpha, rgb)
		if last: return h6
		last, up = stage == 7, stage == 8 and blend
		h7, rgb = self.s7(h6, ws[6] if mix > 7 else mix_w, last, up, alpha, rgb)
		if last: return h7
		last, up = stage == 8, stage == 9 and blend
		h8, rgb = self.s8(h7, ws[7] if mix > 8 else mix_w, last, up, alpha, rgb)
		if last: return h8
		last, up = stage == 9, stage >= 10
		h9, rgb = self.s9(h8, ws[8] if mix > 9 else mix_w, last, up, alpha, rgb)
		if last: return h9
		return rgb

# Generator network
class Generator(Chain):

	def __init__(self, z_size=512, depth=8, channels=(512, 16), max_stage=9):
		super().__init__()
		self.z_size = z_size
		with self.init_scope():
			self.zero = Parameter(Zero(), 1)
			self.one = Parameter(One(), 1)
			self.mapper = FeatureMapper(z_size, depth)
			self.generator = ImageGenerator(z_size, *channels, max_stage)

	def __call__(self, z, stage, alpha=1.0, mix_z=None, mix_stage=None, psi=None, mean_w=None):
		if type(z) is list:
			if psi is None:
				zs = (z + z[-1:] * (stage - 1))[0:stage]
				ws = [self.mapper(z) for z in zs]
				return self.generator(ws, stage, alpha, None if mix_z is None else self.mapper(mix_z), mix_stage)
			else:
				zs = (z + z[-1:] * (stage - 1))[0:stage]
				ws = [self.truncation_trick(self.mapper(z), psi, mean_w) for z in zs]
				return self.generator(ws, stage, alpha, None if mix_z is None else self.truncation_trick(self.mapper(mix_z), psi, mean_w), mix_stage)
		else:
			if psi is None:
				return self.generator(self.mapper(z), stage, alpha, None if mix_z is None else self.mapper(mix_z), mix_stage)
			else:
				return self.generator(self.truncation_trick(self.mapper(z), psi, mean_w), stage, alpha, None if mix_z is None else self.truncation_trick(self.mapper(mix_z), psi, mean_w), mix_stage)

	def resolution(self, stage):
		return (2 * 2 ** stage, 2 * 2 ** stage)

	def generate_latent(self, batch, center=None, sd=1.0):
		zeros = broadcast_to(self.zero, (batch, self.z_size))
		ones = broadcast_to(self.one, (batch, self.z_size))
		ln_var = log(sd ** 2) * ones
		if center is None:
			return gaussian(zeros, ln_var)
		else:
			mean_z = broadcast_to(center, (batch, self.z_size))
			return gaussian(mean_z, ln_var)

	def calculate_mean_w(self, n=10000):
		return mean(self.mapper(self.generate_latent(n)), axis=0)

	def truncation_trick(self, w, psi=0.7, mean_w=None):
		mean_w = mean_w if mean_w is not None else self.calculate_mean_w()
		return mean_w + psi * (w - mean_w)

	def wrap_latent(self, array):
		return Variable(self.xp.array(array))

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

	def wrap_array(self, array):
		return Variable(self.xp.array(array))
