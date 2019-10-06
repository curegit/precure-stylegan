from glob import glob
from os.path import isfile
from chainer.dataset import DatasetMixin
from modules.utilities import load_image, filepath

# Dataset definition for Style GAN
class StyleGanDataset(DatasetMixin):

	def __init__(self, directory, size, preload=False):
		super().__init__()
		self.size = size
		self.preload = preload
		self.images = []
		for e in ["png", "jpg", "jpeg", "gif", "bmp", "tiff"]:
			self.images += [f for f in glob(filepath(directory, "**", e), recursive=True) if isfile(f)]
		if preload:
			self.arraylist = [load_image(i, (size, size)) for i in self.images]

	def __len__(self):
		return len(self.images)

	def get_example(self, index):
		return self.arraylist[index] if self.preload else load_image(self.images[index], (self.size, self.size))
