import sys

from chainer import optimizers
from chainer import serializers
from chainer.iterators import MultiprocessIterator
import chainer.training
from chainer.training import extensions, Trainer

from modules.networks import Generator, Discriminator
from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.utilities import filepath, filerelpath, save_image

from argparse import ArgumentParser

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("datadir", metavar="DATA", help="")
parser.add_argument("-p", "--preload", action="store_true", help="")
parser.add_argument("-d", "--directory", metavar="DEST", default=".", help="")
parser.add_argument("-g", "--generator", metavar="FILE", help="")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="")

parser.add_argument("-b", "--batch", type=int, default=1, help="batch size, affecting memory usage")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-v", "--device", type=int, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# ===== settings =====
directory = filerelpath("/data")
z_size = 512
#batches = [1, 16, 16, 16, 16, 16, 14, 6, 3]
#batch = batches[stage - 1]
preload = False
alpha = 0.0
delta = 0.00005
device = 0

gen = Generator(z_size)
if gend is not None:
	print("Load generator")
	serializers.load_hdf5(gend, gen)

dis = Discriminator()
if disd is not None:
	print("Load discriminator")
	serializers.load_hdf5(disd, dis)

if device >= 0:
	print("Convert to GPU")
	gen.to_gpu()
	dis.to_gpu()

opt_m = optimizers.Adam().setup(gen.mapper)
if optmap is not None:
	print("Load mapper optimizer")
	serializers.load_hdf5(optmap, opt_m)

opt_g = optimizers.Adam().setup(gen.generator)
if optgen is not None:
	print("Load generator optimizer")
	serializers.load_hdf5(optgen, opt_g)

opt_d = optimizers.Adam().setup(dis)
if optdis is not None:
	print("Load discriminator optimizer")
	serializers.load_hdf5(optdis, opt_d)

size = gen.resolution(stage)
#iterator = SerialIterator(StyleGanDataset(directory, size, preload), batch_size=batch, repeat=True, shuffle=True)
iterator = MultiprocessIterator(StyleGanDataset(directory, size, preload), batch_size=batch, repeat=True, shuffle=True, n_processes=16)
updater = StyleGanUpdater(gen, dis, iterator, {"mapper": opt_m, "generator": opt_g, "discriminator": opt_d}, device, stage, alpha, delta)

trainer = Trainer(updater, (epoch, "epoch"), out='results')

def output_image(gen, stage, dir, batch):
	@chainer.training.make_extension()
	def make_image(trainer):
		z = gen.generate_latent(batch)
		y = gen(z, stage, alpha=trainer.updater.alpha)
		y.to_cpu()
		for i in range(batch):
			save_image(y.array[i], filepath(dir, f"{trainer.updater.iteration}_{i}", "png"))
	return make_image


trainer.extend(extensions.LogReport(trigger=(1000, "iteration")))
trainer.extend(extensions.PrintReport(["iteration", "alpha", "loss_gen", "loss_dis"]))
trainer.extend(output_image(gen, stage, "result", 10), trigger=(1000, "iteration"))
trainer.extend(extensions.ProgressBar(update_interval=5))

chainer.global_config.autotune = True

trainer.run()

gen.to_cpu()
dis.to_cpu()

print("Save models")
f = filerelpath("gen.hdf5")
serializers.save_hdf5(f, gen)
f = filerelpath("dis.hdf5")
serializers.save_hdf5(f, dis)

print("Save optimizers")
f = filerelpath("optm.hdf5")
serializers.save_hdf5(f, opt_m)
f = filerelpath("optg.hdf5")
serializers.save_hdf5(f, opt_g)
f = filerelpath("optd.hdf5")
serializers.save_hdf5(f, opt_d)
