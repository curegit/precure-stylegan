import sys
import os
import os.path
import inspect
from PIL import Image
from numpy import asarray, rint, clip, uint8, float32, save, load

# Print to stderr
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

# Wrapper function to do `mkdir -p`
def mkdirp(path):
	os.makedirs(os.path.normpath(path), exist_ok=True)

# Build a file path from a directory path, filename and extension
def filepath(dirpath, filename, ext):
	p = os.path.join(dirpath, filename) + os.extsep + ext
	return os.path.normpath(p)

# Return a complete path of a path relative from the caller's script file
def filerelpath(relpath):
	f = inspect.stack()[1].filename
	d = os.path.dirname(f)
	return os.path.join(d, relpath)

# Return an alternate path if it conflicts
def altfilepath(path, suffix="+"):
	while os.path.lexists(path):
		root, ext = os.path.splitext(path)
		head, tail = os.path.split(root)
		path = os.path.join(head, tail + suffix) + ext
	return path

# Deserialize a Numpy array
def load_array(path):
	return load(path)

# Serialize a Numpy array
def save_array(array, path):
	save(path, array)

# Load an image and return it as a Numpy array
def load_image(path, size=None):
	img = Image.open(path).convert("RGB")
	if size is not None: img = img.resize(size, Image.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1).astype(float32) / 255

# Convert a Numpy array to a Pillow image
def array2image(array):
	array = clip(rint(array * 255), 0, 255).astype(uint8).transpose(1, 2, 0)
	return Image.fromarray(array, "RGB")

# Save a Numpy array as an image
def save_image(array, path):
	array2image(array).save(path)
