from datetime import datetime
from argparse import ArgumentParser
from chainer import global_config
from chainer import optimizers
from chainer import serializers
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions, Trainer, make_extension
from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.networks import Generator, Discriminator
from modules.utilities import mkdirp, filepath, altfilepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("datadir", metavar="DATA", help="")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
parser.add_argument("-r", "--result", metavar="DEST", default="result", help="")
parser.add_argument("-g", "--generator", metavar="FILE", help="")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="optimizers of mapper, generator and discriminator")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="growth stage to train")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-b", "--batch", type=int, default=4, help="batch size, affecting memory usage")
parser.add_argument("-e", "--epoch", type=int, default=1, help="")
parser.add_argument("-a", "--alpha", type=float, default=0.0, help="")
parser.add_argument("-t", "--delta", type=float, default=2**-9, help="")
parser.add_argument("-v", "--device", type=int, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Validate arguments
batch = max(1, args.batch)
epoch = max(1, args.epoch)
alpha = max(0.0, min(1.0, args.alpha))
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
iterator = MultiprocessIterator(dataset, batch_size=batch, repeat=True, shuffle=True, n_prefetch=4)
n = dataset.length()
if n < 1:
	print("No image found in dataset directory")
	exit(1)

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
print("Initializing optimizers")
mapper_optimizer = optimizers.Adam().setup(generator.mapper)
generator_optimizer = optimizers.Adam().setup(generator.generator)
discriminator_optimizer = optimizers.Adam().setup(discriminator)

# Load optimizers
if args.optimizers is not None:
	print("Loading mapper's optimizer")
	serializers.load_hdf5(args.optimizers[0], mapper_optimizer)
	print("Loading generator's optimizer")
	serializers.load_hdf5(args.optimizers[1], generator_optimizer)
	print("Loading discriminator's optimizer")
	serializers.load_hdf5(args.optimizers[2], discriminator_optimizer)

# Config chainer
global_config.autotune = True

# Prepare updater
updater = StyleGanUpdater(generator, discriminator, iterator, {"mapper": mapper_optimizer, "generator": generator_optimizer, "discriminator": discriminator_optimizer}, device, args.stage, alpha, delta)

# Init result directory
mkdirp(args.result)

#
def save_middle_images(generator, stage, directory, number, batch):
	@make_extension()
	def func(trainer):
		c = 0
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			y = generator(z, args.stage)
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{trainer.updater.iteration}_{c + i + 1}", "png")
				save_image(y.array[i], path)
			c += n
	return func

# Prepare trainer
trainer = Trainer(updater, (epoch, "epoch"), out=args.result)
trainer.extend(extensions.ProgressBar(update_interval=5))
trainer.extend(extensions.LogReport(trigger=(1000, "iteration")))
trainer.extend(extensions.PrintReport(["iteration", "alpha", "loss (gen)", "loss (dis)"]))
trainer.extend(extensions.PlotReport(["alpha", "loss (gen)", "loss (dis)"], "iteration", trigger=(10, "iteration"), file_name="loss.png"))
trainer.extend(save_middle_images(generator, args.stage, args.result, 20, batch), trigger=(1000, "iteration"))

# Run ML
trainer.run()

# Save models
t = datetime.now().strftime("%y%m%d%H")
print("Saving models")
generator.to_cpu()
discriminator.to_cpu()
serializers.save_hdf5(altfilepath(filepath(args.result, f"gen{t}", "hdf5")), generator)
serializers.save_hdf5(altfilepath(filepath(args.result, f"dis{t}", "hdf5")), discriminator)

# Save optimizers
print("Saving optimizers")
serializers.save_hdf5(altfilepath(filepath(args.result, f"mopt{t}", "hdf5")), mapper_optimizer)
serializers.save_hdf5(altfilepath(filepath(args.result, f"gopt{t}", "hdf5")), generator_optimizer)
serializers.save_hdf5(altfilepath(filepath(args.result, f"dopt{t}", "hdf5")), discriminator_optimizer)
