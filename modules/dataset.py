from glob import glob, escape
from os.path import isfile, join
from chainer.dataset import DatasetMixin
from modules.utilities import load_image, filepath

# Dataset abstraction of images in a folder and its subfolders
class StyleGanDataset(DatasetMixin):

	def __init__(self, directory, size, preload=False):
		super().__init__()
		self.size = size
		self.preload = preload
		self.images = []
		for e in ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"]:
			pattern = filepath(escape(directory), join("**", "*"), e)
			self.images += [f for f in glob(pattern, recursive=True) if isfile(f)]
		if preload:
			self.arrays = [load_image(i, size) for i in self.images]

	def __len__(self):
		return len(self.images)

	def length(self):
		return self.__len__()

	def get_example(self, index):
		return self.arrays[index] if self.preload else load_image(self.images[index], self.size)
