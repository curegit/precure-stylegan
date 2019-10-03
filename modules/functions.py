from numpy import float32
from numpy.random import normal

def normal_random(shape, mean=0.0, stddev=1.0):
	return normal(mean, stddev, size=shape).astype(float32)
