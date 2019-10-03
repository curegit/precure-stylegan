from chainer import Chain
from chainer.links import Parameter, Scale
from chainer.functions import broadcast_to
from chainer.initializers import Zero, One
from modules.links import EqualizedLinear, EqualizedConvolution2D

#
class NoiseAdder(Chain):

	def __init__(self, ch):
		super().__init__()
		with self.init_scope():
			# shape? correct
			self.s = Scale(W_shape=(ch))

	def __call__(self, x):
		# noise: (batch, w, h)
		noise = self.xp.random.normal(size=(x.shape[0], 1, x.shape[2], x.shape[3])).astype(np.float32)
		n = broadcast_to(noise, x.shape)
		#s = broadcast_to(self.s(), x.shape)
		return x + self.s(n)

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
			self.p1 = Parameter()
			self.n1 = NoiseAdder()
			self.a1 = StyleAffineTransform()
			self.i1 = AdaptiveInstanceNormalization()
			self.c1 = EqualizedConvolution2D()
			self.n2 = NoiseAdder()
			self.a2 = StyleAffineTransform()
			self.i2 = AdaptiveInstanceNormalization()

	def __call__(self, x, w)


#
class SynthesisNetwork(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D()
			self.n1 = NoiseAdder()
			self.a1 = StyleAffineTransform()
			self.i1 = AdaptiveInstanceNormalization()
			self.c2 = EqualizedConvolution2D()
			self.n2 = NoiseAdder()
			self.a2 = StyleAffineTransform()
			self.i2 = AdaptiveInstanceNormalization()

	def __call__(self, x, w):

