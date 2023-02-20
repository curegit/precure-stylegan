import sys
import os
import os.path
import cv2
from os.path import join
from glob import glob, escape
from PIL import Image

min_size = 64

def mkdirp(path):
	os.makedirs(os.path.normpath(path), exist_ok=True)

def relpath(relpath):
	return join(os.path.dirname(__file__), relpath)

cascade = cv2.CascadeClassifier(relpath("lbpcascade_animeface.xml"))

def detect(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	return cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3, minSize=(min_size, min_size))

def detect_test(image, faces, dest):
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imwrite(dest, image)

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
	ext_y2 = round(h / 2 * 0.10)
	desy1 = y - ext_y1
	desy2 = y + h + ext_y2
	overy1 = max(0, 0 - desy1)
	overy2 = max(0, desy2 - img.height)
	clamy1 = desy1 + overy1
	clamy2 = desy2 - overy2
	c1 = img.crop((clamx1, clamy1, clamx2, clamy2))
	return crop_max_square(c1)

def crop(src, dest):
	fs = []
	fs += glob(join(escape(src), "**/*.png"), recursive=True)
	fs += glob(join(escape(src), "**/*.jpg"), recursive=True)
	fs += glob(join(escape(src), "**/*.jpeg"), recursive=True)
	fs += glob(join(escape(src), "**/*.gif"), recursive=True)
	fs += glob(join(escape(src), "**/*.bmp"), recursive=True)
	fs += glob(join(escape(src), "**/*.webp"), recursive=True)
	mkdirp(join(dest, "test"))
	mkdirp(join(dest, "crop"))

	for i, f in enumerate(fs):
		subd = str(i // 1000)
		mkdirp(join(dest, f"test/{subd}"))
		image = cv2.imread(f, cv2.IMREAD_COLOR)
		faces = detect(image)
		detect_test(image, faces, join(dest, f"test/{subd}/{i}.png"))
		img = Image.open(f)
		mkdirp(join(dest, f"crop/{subd}"))
		for j, face in enumerate(faces):
			fi = crop_face(img, *face)
			k = f"_{j}" if j > 0 else ""
			fi.save(join(dest, f"crop/{subd}/{i}{k}.png"))

if len(sys.argv) < 3:
	print("Crop square by images")
	print("usage: crop.py SRCDIR DESTDIR")
else:
	crop(sys.argv[1], sys.argv[2])
