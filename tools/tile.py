import sys
import itertools
from PIL import Image

def tile(x, y, imgs):
	imgs = iter(imgs)
	img = Image.open(next(imgs))
	w, h = img.size
	canvas = Image.new("RGB", (w * x, h * y), "white")
	for j, i in itertools.product(range(y), range(x)):
		canvas.paste(img, (i * w, j * h))
		img = next(imgs, None)
		if img is None:
			break
		img = Image.open(img).convert("RGB")
	return canvas

if len(sys.argv) < 4:
	print("Create a N by M tiled image matrix")
	print("usage: tile.py M N FILE [FILE ...] OUTPUT")
else:
	tile(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:-1]).save(sys.argv[-1])
