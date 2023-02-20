import sys
import os
import os.path
import cv2
from os.path import join
from glob import glob, escape
from PIL import Image

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def mkdirp(path):
	os.makedirs(os.path.normpath(path), exist_ok=True)

def relpath(relpath):
	return join(os.path.dirname(__file__), relpath)

cascade = cv2.CascadeClassifier(relpath("lbpcascade_animeface.xml"))

def detect(cv_image, min_size):
	gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
	eq_gray = cv2.equalizeHist(gray)
	return cascade.detectMultiScale(eq_gray, scaleFactor=1.02, minNeighbors=3, minSize=(min_size, min_size))

def mark_detect(cv_image, faces, dest):
	for (x, y, w, h) in faces:
		cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imwrite(dest, cv_image)

def crop_center(pil_img, crop_width, crop_height):
	img_width, img_height = pil_img.size
	return pil_img.crop((
		(img_width - crop_width) // 2,
		(img_height - crop_height) // 2,
		(img_width + crop_width) // 2,
		(img_height + crop_height) // 2,
	))

def crop_max_square(pil_img):
	return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def crop_face(img, x, y, w, h):
	ext_x = round(w / 2 * 0.37759)
	desx1 = x - ext_x
	desx2 = x + w + ext_x
	overx1 = max(0, 0 - desx1)
	overx2 = max(0, desx2 - img.width)
	overx = max(overx1, overx2)
	clamx1 = desx1 + overx
	clamx2 = desx2 - overx
	ext_y1 = round(h / 2 * 0.72457)
	ext_y2 = round(h / 2 * 0.1)
	desy1 = y - ext_y1
	desy2 = y + h + ext_y2
	overy1 = max(0, 0 - desy1)
	overy2 = max(0, desy2 - img.height)
	clamy1 = desy1 + overy1
	clamy2 = desy2 - overy2
	c1 = img.crop((clamx1, clamy1, clamx2, clamy2))
	return crop_max_square(c1)

def crop(src, dest, min_size=64):
	fs = []
	fs += glob(join(escape(src), "**/*.png"), recursive=True)
	fs += glob(join(escape(src), "**/*.jpg"), recursive=True)
	fs += glob(join(escape(src), "**/*.jpeg"), recursive=True)
	fs += glob(join(escape(src), "**/*.tif"), recursive=True)
	fs += glob(join(escape(src), "**/*.tiff"), recursive=True)
	fs += glob(join(escape(src), "**/*.bmp"), recursive=True)
	fs += glob(join(escape(src), "**/*.webp"), recursive=True)
	mkdirp(join(dest, "test"))
	mkdirp(join(dest, "crop"))
	for i, f in enumerate(fs):
		subd = str(i // 1000)
		mkdirp(join(dest, f"test/{subd}"))
		image = cv2.imread(f, cv2.IMREAD_COLOR)
		if image is None:
			eprint(f"Skipped an unreadable file: '{f}'")
			continue
		faces = detect(image, min_size=min_size)
		mark_detect(image, faces, join(dest, f"test/{subd}/{i}.png"))
		img = Image.open(f)
		mkdirp(join(dest, f"crop/{subd}"))
		for j, face in enumerate(faces):
			sq_img = crop_face(img, *face)
			sq_img.save(join(dest, f"crop/{subd}/{i}_{j}.png"))

if len(sys.argv) == 3:
	crop(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 4:
	crop(sys.argv[1], sys.argv[2], min_size=int(sys.argv[3]))
else:
	print("Crop square facial images from Anime image set")
	print("usage: crop.py SRC_DIR DEST_DIR [MIN_SIZE]")
