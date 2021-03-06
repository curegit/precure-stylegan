from chainer import Parameter, Chain
from chainer.links import Scale
from chainer.functions import mean, sqrt, broadcast_to, resize_images, gaussian
from chainer.initializers import Zero, One
from modules.links import EqualizedLinear, EqualizedConvolution2D, LeakyReluLink, LerpBlendLink

# Link that returns constant value
class Constant(Chain):

	def __init__(self, channels, height, width):
		super().__init__()
		with self.init_scope():
			self.p = Parameter(One(), (channels, height, width))

	def __call__(self, batch):
		return broadcast_to(self.p, (batch, *self.p.shape))

# 2x upsample operation as link
class Upsampler(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		height, width = x.shape[2:]
		return resize_images(x, (height * 2, width * 2), align_corners=False)

# Noise injection layer
class NoiseAdder(Chain):

	def __init__(self, channels):
		super().__init__()
		with self.init_scope():
			self.z = Parameter(Zero(), 1)
			self.s = Scale(W_shape=channels)

	def __call__(self, x):
		n = self.generate_noises(*x.shape)
		return x + self.s(n)

	def generate_noises(self, batch, channels, height, width):
		z = broadcast_to(self.z, (batch, 1, height, width))
		z.unchain_backward()
		return broadcast_to(gaussian(z, z), (batch, channels, height, width))

# Learnable transform from W to style
class StyleAffineTransform(Chain):

	def __init__(self, in_size, out_size):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(in_size, out_size, initial_bias=One(), gain=1)
			self.b = EqualizedLinear(in_size, out_size, initial_bias=Zero(), gain=1)

	def __call__(self, w):
		return self.s(w), self.b(w)

# AdaIN layer for applying style
class AdaptiveInstanceNormalization(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x, ys, yb):
		s = broadcast_to(ys.reshape(ys.shape + (1, 1)), ys.shape + x.shape[2:])
		b = broadcast_to(yb.reshape(yb.shape + (1, 1)), yb.shape + x.shape[2:])
		e = x - broadcast_to(mean(x, axis=1, keepdims=True), x.shape)
		sd = broadcast_to(sqrt(mean(e ** 2, axis=1, keepdims=True) + 1e-8), x.shape)
		return s * e / sd + b

# Pixelwise feature vector normalization
class PixelwiseFeatureMapNormalization(Chain):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		return x / sqrt(mean(x ** 2, axis=1, keepdims=True) + 1e-8)

# First block of image generator
class InitialSynthesisNetwork(Chain):

	def __init__(self, in_channels, out_channels, w_size, height, width):
		super().__init__()
		with self.init_scope():
			self.p1 = Constant(in_channels, height, width)
			self.n1 = NoiseAdder(in_channels)
			self.f1 = PixelwiseFeatureMapNormalization()
			self.a1 = StyleAffineTransform(w_size, in_channels)
			self.i1 = AdaptiveInstanceNormalization()
			self.c1 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, reflect=True)
			self.n2 = NoiseAdder(out_channels)
			self.r1 = LeakyReluLink(0.2)
			self.f2 = PixelwiseFeatureMapNormalization()
			self.a2 = StyleAffineTransform(w_size, out_channels)
			self.i2 = AdaptiveInstanceNormalization()
			self.us = Upsampler()
			self.rgb = EqualizedConvolution2D(out_channels, 3, ksize=1, stride=1, pad=0, gain=1)

	def __call__(self, w, last=False, upsample=False):
		h1 = self.p1(w.shape[0])
		h2 = self.n1(h1)
		h3 = self.f1(h2)
		ys1, yb1 = self.a1(w)
		h4 = self.i1(h3, ys1, yb1)
		h5 = self.c1(h4)
		h6 = self.n2(h5)
		h7 = self.r1(h6)
		h8 = self.f2(h7)
		ys2, yb2 = self.a2(w)
		h9 = self.i2(h8, ys2, yb2)
		if last:
			return self.rgb(h9), None
		else:
			up = self.us(h9)
			return up, self.rgb(up) if upsample else None

# Tail blocks of image generator
class SynthesisNetwork(Chain):

	def __init__(self, in_channels, out_channels, w_size):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, reflect=True)
			self.n1 = NoiseAdder(out_channels)
			self.r1 = LeakyReluLink(0.2)
			self.f1 = PixelwiseFeatureMapNormalization()
			self.a1 = StyleAffineTransform(w_size, out_channels)
			self.i1 = AdaptiveInstanceNormalization()
			self.c2 = EqualizedConvolution2D(out_channels, out_channels, ksize=3, stride=1, pad=1, reflect=True)
			self.n2 = NoiseAdder(out_channels)
			self.r2 = LeakyReluLink(0.2)
			self.f2 = PixelwiseFeatureMapNormalization()
			self.a2 = StyleAffineTransform(w_size, out_channels)
			self.i2 = AdaptiveInstanceNormalization()
			self.us = Upsampler()
			self.lb = LerpBlendLink()
			self.rgb = EqualizedConvolution2D(out_channels, 3, ksize=1, stride=1, pad=0, gain=1)

	def __call__(self, x, w, last=False, upsample=False, alpha=1.0, blend=None):
		h1 = self.c1(x)
		h2 = self.n1(h1)
		h3 = self.r1(h2)
		h4 = self.f1(h3)
		ys1, yb1 = self.a1(w)
		h5 = self.i1(h4, ys1, yb1)
		h6 = self.c2(h5)
		h7 = self.n2(h6)
		h8 = self.r2(h7)
		h9 = self.f2(h8)
		ys2, yb2 = self.a2(w)
		h10 = self.i2(h9, ys2, yb2)
		if last:
			rgb = self.rgb(h10)
			return rgb if blend is None else self.lb(blend, rgb, alpha), None
		else:
			up = self.us(h10)
			return up, self.rgb(up) if upsample else None
