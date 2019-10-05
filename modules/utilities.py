import os
import os.path as path
import inspect

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
	return os.getcwd() if f == "<stdin>" else path.join(path.dirname(f), relpath)
