from sys import float_info
from os.path import basename

# 非負整数を受け入れる変換関数
def uint(str):
	value = int(str)
	if value >= 0:
		return value
	raise ValueError()

# 正の整数を受け入れる変換関数
def natural(str):
	value = int(str)
	if value > 0:
		return value
	raise ValueError()

# 非負実数を受け入れる変換関数
def ufloat(str):
	value = float(str)
	if value >= 0:
		return value
	raise ValueError()

# 正の実数を受け入れる変換関数
def positive(str):
	value = float(str)
	if value >= float_info.epsilon:
		return value
	raise ValueError()

# 0-1 の実数を受け入れる変換関数
def rate(str):
	value = float(str)
	if 0 <= value <= 1:
		return value
	raise ValueError()

# ファイル名を受け入れる変換関数
def filename(str):
	if str == basename(str):
		return str
	raise ValueError()

# デバイス ID またはデバイス種別の文字列を受け入れる変換関数
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
