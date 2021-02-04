from sys import float_info
from os.path import basename

# Converter to a non-negative integer
def uint(str):
	value = int(str)
	if value >= 0:
		return value
	raise ValueError()

# Converter to a positive integer
def natural(str):
	value = int(str)
	if value > 0:
		return value
	raise ValueError()

# Converter to a non-negative decimal
def ufloat(str):
	value = float(str)
	if value >= 0:
		return value
	raise ValueError()

# Converter to a positive decimal
def positive(str):
	value = float(str)
	if value >= float_info.epsilon:
		return value
	raise ValueError()

# Converter to a decimal in the range 0-1
def rate(str):
	value = float(str)
	if 0 <= value <= 1:
		return value
	raise ValueError()

# Type checker for filenames
def filename(str):
	if str == basename(str):
		return str
	raise ValueError()

# Converter to a CUDA device ID
def device(str):
	value = str.lower()
	if value == "cpu":
		return -1
	if value == "gpu":
		return 0
	value = int(str)
	if value >= -1:
		return value
	raise ValueError()
