import os
import os.path as path
import inspect
from PIL import Image
from numpy import asarray, rint, clip, uint8, float32
from chainer import Variable

# Wrapper for doing mkdir -p
def mkdirp(path):
	os.makedirs(path, exist_ok=True)

# Build filepath from dirpath, filename and extension
def filepath(dirpath, filename, ext):
	p = path.join(dirpath, filename) + os.extsep + ext
	return path.normpath(p)

# Build path relatively from caller's script directory
def filerelpath(relpath):
	f = inspect.stack()[1].filename
	d = os.getcwd() if f == "<stdin>" else path.dirname(f)
	return path.join(d, relpath)

# Load image to return chainer variable
def load_image(path, size=None):
	img = Image.open(path)
	if size is not None: img = img.resize(size, Image.LANCZOS)
	array = asarray(img, dtype=uint8).transpose(2, 0, 1).astype(float32) / 255
	return Variable(array)

# Save chainer variable of 2d array as image
def save_image(variable, path):
	variable.to_cpu()
	array = clip(rint(variable.array * 255), 0, 255).astype(uint8).transpose(1, 2, 0)
	img = Image.fromarray(array, "RGB")
	img.save(path)
