import os
import os.path as path
import inspect
from PIL import Image
from numpy import asarray, rint, clip, uint8, float32

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

# Make alternate file path
def altfilepath(fpath):
	while path.lexists(fpath):
		root, ext = path.splitext(fpath)
		head, tail = path.split(root)
		fpath = head + "_" + tail + ext
	return fpath

# Load image to return numpy array
def load_image(path, size=None):
	img = Image.open(path)
	if size is not None: img = img.resize(size, Image.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1).astype(float32) / 255

# Save 2d numpy array as image
def save_image(array, path):
	array = clip(rint(array * 255), 0, 255).astype(uint8).transpose(1, 2, 0)
	img = Image.fromarray(array, "RGB")
	img.save(path)
