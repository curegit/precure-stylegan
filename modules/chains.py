from chainer import Chain, Parameter
from chainer.links import Scale
from chainer.functions import broadcast_to
from chainer.initializers import Zero, One
from modules.links import EqualizedLinear, EqualizedConvolution2D
from modules.functions import normal_random

# Link that returns constant value
class Constant(Chain):

	def __init__(self, channels, width, height):
		super().__init__()
		with self.init_scope():
			self.p = Parameter(One(), (channels, width, height))

	def __call__(self):
		return self.p

# Noise injection layer
class NoiseAdder(Chain):

	def __init__(self, channels):
		super().__init__()
		with self.init_scope():
			self.s = Scale(W_shape=(channels))

	def __call__(self, x):
		n = self.generate_noises(*x.shape)
		return x + self.s(n)

	def generate_noises(batch, channels, height, width):
		n = normal_random(shape=(batch, 1, height, width))
		return broadcast_to(n, (batch, channels, height, width))

#
class StyleAffineTransform(Chain):

	def __init__(self, in_size, out_size):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(in_size, out_size, initial_bias=One())
			self.b = EqualizedLinear(in_size, out_size, initial_bias=Zero())

	def __call__(self, w):
		return self.s(w), self.b(w)

#
class AdaptiveInstanceNormalization(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x, ys, yb):
		m = F.broadcast_to(F.mean(x, axis=1, keepdims=True), x.shape)
		v = F.mean((x - m) * (x - m), axis=1, keepdims=True)
		s = F.broadcast_to(F.sqrt(v + 1e-8), x.shape)
		return style_s * (x - m) / s + style_b

#
class FirstSynthesisNetwork(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.p1 = Constant()
			self.n1 = NoiseAdder()
			self.a1 = StyleAffineTransform()
			self.i1 = AdaptiveInstanceNormalization()
			self.c1 = EqualizedConvolution2D()
			self.n2 = NoiseAdder()
			self.a2 = StyleAffineTransform()
			self.i2 = AdaptiveInstanceNormalization()
			#self.rgb =

	def __call__(self, x, w):



#
class SynthesisNetwork(Chain):

	def __init__(self, in_channels, out_channels, w_size):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D()
			self.n1 = NoiseAdder()
			self.a1 = StyleAffineTransform(z_size)
			self.i1 = AdaptiveInstanceNormalization()
			self.c2 = EqualizedConvolution2D()
			self.n2 = NoiseAdder()
			self.a2 = StyleAffineTransform()
			self.i2 = AdaptiveInstanceNormalization()
			#self.rgb =

	def __call__(self, x, w):

