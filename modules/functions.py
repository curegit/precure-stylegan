from numpy import float32
from numpy.random import normal
from chainer import Variable
from chainer.functions import resize_images

# Generate normally-distributed random array
def normal_random(shape, mean=0.0, stddev=1.0):
	return Variable(normal(mean, stddev, size=shape).astype(float32))

# 1/2 image reduction by bilinear
def shrink_images(x):
	h, w = x.shape[2:]
	return resize_images(x, (h // 2, w // 2))

# 2x image extension by bilinear
def enlarge_images(x):
	h, w = x.shape[2:]
	return resize_images(x, (h * 2, w * 2))

# Clamped image interpolation
def lerp_blend(x, y, alpha=0.5):
	if alpha <= 0.0: return x
	if alpha >= 1.0: return y
	return (1 - alpha) * x + alpha * y
