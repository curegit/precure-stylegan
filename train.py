from argparse import ArgumentParser

from chainer import optimizers
from chainer import serializers
from chainer.iterators import MultiprocessIterator
import chainer.training
from chainer.training import extensions, Trainer

from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.networks import Generator, Discriminator
from modules.utilities import filepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("datadir", metavar="DATA", help="")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
#parser.add_argument("-d", "--directory", metavar="DEST", default="result", help="")
parser.add_argument("-g", "--generator", metavar="FILE", help="")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="optimizers of mapper, generator and discriminator")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="growth stage to train")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-b", "--batch", type=int, default=4, help="batch size, affecting memory usage")
parser.add_argument("-e", "--epoch", type=int, default=1, help="")
parser.add_argument("-a", "--alpha", type=float, default=0.0, help="")
parser.add_argument("-t", "--delta", type=float, default=2**-16, help="")
parser.add_argument("-v", "--device", type=int, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Validate arguments
batch = max(1, args.batch)
epoch = max(1, args.epoch)
alpha = max(0.0, args.alpha)
delta = max(0.0, args.delta)
size = max(1, args.size)
depth = max(1, args.mlp)
device = max(-1, args.device)

# Init models
print("Initializing models")
generator = Generator(size, depth)
discriminator = Discriminator()

# Prepare dataset
h, w = generator.resolution(args.stage)
dataset = StyleGanDataset(args.datadir, (w, h), args.preload)
iterator = MultiprocessIterator(dataset, batch_size=batch, repeat=True, shuffle=True, n_processes=16)
n = dataset.__len__()

# Print information
print(f"Epoch: {epoch}, Batch: {batch}, Images: {n}")
print(f"MLP: {size}x{depth}, Stage: {args.stage} ({w}x{h})")
print(f"Device: {'CPU' if device < 0 else f'GPU {device}'}")

# Load models
if args.generator is not None:
	print("Loading generator")
	serializers.load_hdf5(args.generator, generator)
if args.discriminator is not None:
	print("Loading discriminator")
	serializers.load_hdf5(args.discriminator, discriminator)

# GPU setting
if device >= 0:
	print("Converting to GPU")
	generator.to_gpu(device)
	discriminator.to_gpu(device)

# Init optimizers
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

# Load optimizers

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
