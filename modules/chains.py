from chainer import Chain
from chainer.initializers import Zero, One
from modules.links import EqualizedLinear, EqualizedConvolution2D

#
class NoiseAdder(Chain):

	def __init__(self, ch):
		super().__init__()
		with self.init_scope():
			self.s = Const(init.Zero(), (1, ch, 1, 1))

	def __call__(self, x, noise=None):
		# ?
		z = F.broadcast_to(z, x.shape)
		s = F.broadcast_to(self.s(), x.shape)
		return x + s * z

	def generate_noise(self, batch, width, height):
		# ?
		return self.xp.random.normal(size=(x.shape[0], 1, x.shape[2], x.shape[3])).astype(np.float32)

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
class AdaptiveInstanceNormalization(Link):

	def __init__(self):
		super().__init__()

	def __call__(self, x, ys, yb):


#
class SynthesisNetwork(Chain):
