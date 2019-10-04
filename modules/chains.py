from chainer import Chain, Parameter
from chainer.links import Scale
from chainer.functions import mean, sqrt, broadcast_to, unpooling_2d
from chainer.initializers import Zero, One
from modules.links import EqualizedLinear, EqualizedConvolution2D
from modules.functions import normal_random

# Link that returns constant value
class Constant(Chain):

	def __init__(self, channels, height, width):
		super().__init__()
		with self.init_scope():
			self.p = Parameter(One(), (channels, height, width))

	def __call__(self, batch):
		return broadcast_to(self.p, (batch,) + self.p.shape)

# 2x Upsample operation as link
class Upsampler(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		height, width = x.shape[2:]
		return unpooling_2d(x, ksize=2, stride=2, pad=0, outsize=(height * 2, width * 2))

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

# Learnable transform from W to style
class StyleAffineTransform(Chain):

	def __init__(self, in_size, out_size):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(in_size, out_size, initial_bias=One())
			self.b = EqualizedLinear(in_size, out_size, initial_bias=Zero())

	def __call__(self, w):
		return self.s(w), self.b(w)

# AdaIN layer for applying style
class AdaptiveInstanceNormalization(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x, ys, yb):
		m = broadcast_to(mean(x, axis=1, keepdims=True), x.shape)
		sd = broadcast_to(sqrt(mean((x - m) ** 2, axis=1, keepdims=True) + 1e-8), x.shape)
		return ys * (x - m) / sd + yb

# First block of image generator
class InitialSynthesisNetwork(Chain):

	def __init__(self, in_channels, out_channels, w_size, height, width):
		super().__init__()
		with self.init_scope():
			self.p1 = Constant(in_channels, height, width)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = StyleAffineTransform(w_size, out_channels)
			self.i1 = AdaptiveInstanceNormalization()
			self.c1 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.n2 = NoiseAdder(out_channels)
			self.a2 = StyleAffineTransform(w_size, out_channels)
			self.i2 = AdaptiveInstanceNormalization()
			self.rgb = EqualizedConvolution2D(out_channels, 3, ksize=1, stride=1, pad=0)

	def __call__(self, w, last=False):
		h1 = self.p1(w.shape[0])
		h2 = self.n1(h1)
		ys1, yb1 = self.a1(w)
		h3 = self.i1(h2, ys1, yb1)
		h4 = self.c1(h3)
		h5 = self.n2(h4)
		ys2, yb2 = self.a2(w)
		h6 = self.i2(h5, ys2, yb2)
		return self.rgb(h6) if last else h6

# Tail blocks of image generator
class SynthesisNetwork(Chain):

	def __init__(self, in_channels, out_channels, w_size):
		super().__init__()
		with self.init_scope():
			self.u1 = Upsampler()
			self.c1 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = StyleAffineTransform(w_size, out_channels)
			self.i1 = AdaptiveInstanceNormalization()
			self.c2 = EqualizedConvolution2D(out_channels, out_channels, ksize=3, stride=1, pad=1)
			self.n2 = NoiseAdder(out_channels)
			self.a2 = StyleAffineTransform(w_size, out_channels)
			self.i2 = AdaptiveInstanceNormalization()
			self.rgb = EqualizedConvolution2D(out_channels, 3, ksize=1, stride=1, pad=0)

	def __call__(self, x, w, last=False):
		h1 = self.u1(x)
		h2 = self.c1(h1)
		h3 = self.n1(h2)
		ys1, yb1 = self.a1(w)
		h4 = self.i1(h3, ys1, yb1)
		h5 = self.c2(h4)
		h6 = self.n2(h5)
		ys2, yb2 = self.a2(w)
		h7 = self.i2(h6, ys2, yb2)
		return self.rgb(h7) if last else h6
