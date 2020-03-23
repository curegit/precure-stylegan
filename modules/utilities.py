import os
import os.path
import inspect
from PIL import Image
from numpy import asarray, rint, clip, uint8, float32, save, load

# Wrapper for doing mkdir -p
def mkdirp(path):
	os.makedirs(path, exist_ok=True)

# Build filepath from dirpath, filename and extension
def filepath(dirpath, filename, ext):
	p = os.path.join(dirpath, filename) + os.extsep + ext
	return os.path.normpath(p)

# Build path relatively from caller's script directory
def filerelpath(relpath):
	f = inspect.stack()[1].filename
	d = os.getcwd() if f == "<stdin>" else os.path.dirname(f)
	return os.path.join(d, relpath)

# Make alternate file path
def altfilepath(path):
	while os.path.lexists(path):
		root, ext = os.path.splitext(path)
		head, tail = os.path.split(root)
		path = os.path.join(head, "_" + tail) + ext
	return path

# Load numpy array from binary file
def load_array(path):
	return load(path)

# Save numpy array in binary file
def save_array(array, path):
	save(path, array)

# Load image to return numpy array
def load_image(path, size=None):
	img = Image.open(path).convert("RGB")
	if size is not None: img = img.resize(size, Image.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1).astype(float32) / 255

# Convert 2d numpy array to image
def array2image(array):
	array = clip(rint(array * 255), 0, 255).astype(uint8).transpose(1, 2, 0)
	return Image.fromarray(array, "RGB")

# Save 2d numpy array as image
def save_image(array, path):
	array2image(array).save(path)
