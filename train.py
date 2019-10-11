from os.path import basename
from datetime import datetime
from argparse import ArgumentParser
from chainer import optimizers, serializers, global_config
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer, extensions, make_extension
from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.networks import Generator, Discriminator
from modules.utilities import mkdirp, filepath, altfilepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("dataset", metavar="DATA", help="dataset directory which stores images")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
parser.add_argument("-c", "--current", action="store_true", help="save completed models in current directory")
parser.add_argument("-x", "--no-datetime", dest="nodate", action="store_true", help="don't add datetime prefix to model files")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-r", "--result", metavar="DEST", default="results", help="destination directory for models, logs, middle images, and so on")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained generator to load and retrain")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="HDF5 file of serialized trained discriminator to load and retrain")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="Snapshot of optimizers of mapper, generator, and discriminator")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="growth stage to train")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-b", "--batch", type=int, default=4, help="batch size, affecting memory usage")
parser.add_argument("-e", "--epoch", type=int, default=1, help="")
parser.add_argument("-a", "--alpha", type=float, default=0.0, help="")
parser.add_argument("-t", "--delta", type=float, default=2**-8, help="")
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
dataset = StyleGanDataset(args.dataset, (w, h), args.preload)
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

# Define extension to output images in progress
def save_middle_images(generator, stage, directory, number, batch):
	@make_extension()
	def func(trainer):
		c = 0
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			y = generator(z, stage)
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{c + i + 1}", "png")
				save_image(y.array[i], path)
			c += n
	return func

# Prepare trainer
trainer = Trainer(updater, (epoch, "epoch"), out=args.result)
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.extend(extensions.LogReport(trigger=(1000, "iteration")))
trainer.extend(extensions.PrintReport(["iteration", "alpha", "loss (gen)", "loss (dis)"]))
trainer.extend(extensions.PlotReport(["alpha", "loss (gen)", "loss (dis)"], "iteration", trigger=(400, "iteration"), file_name="loss.png"))
trainer.extend(save_middle_images(generator, args.stage, args.result, 10, batch), trigger=(1000, "iteration"))

# Run ML
trainer.run()

# Save models
print("Saving models")
generator.to_cpu()
discriminator.to_cpu()
t = datetime.now().strftime("%m%d%H")
gname = f"gen{'' if args.nodate else t}"
dname = f"dis{'' if args.nodate else t}"
gpath = filepath("." if args.current else args.result, gname, "hdf5")
dpath = filepath("." if args.current else args.result, dname, "hdf5")
gpath = gpath if args.force else altfilepath(gpath)
dpath = dpath if args.force else altfilepath(dpath)
serializers.save_hdf5(gpath, generator)
print(f"Generator saved as {basename(gpath)}")
serializers.save_hdf5(dpath, discriminator)
print(f"Discriminator saved as {basename(dpath)}")

# Save optimizers
print("Saving optimizers")
omname = f"mopt{'' if args.nodate else t}"
ogname = f"gopt{'' if args.nodate else t}"
odname = f"dopt{'' if args.nodate else t}"
ompath = filepath("." if args.current else args.result, omname, "hdf5")
ogpath = filepath("." if args.current else args.result, ogname, "hdf5")
odpath = filepath("." if args.current else args.result, odname, "hdf5")
ompath = ompath if args.force else altfilepath(ompath)
ogpath = ogpath if args.force else altfilepath(ogpath)
odpath = odpath if args.force else altfilepath(odpath)
serializers.save_hdf5(ompath, mapper_optimizer)
serializers.save_hdf5(ogpath, generator_optimizer)
serializers.save_hdf5(odpath, discriminator_optimizer)
print(f"Optimizers saved as {basename(ompath)}, {basename(ogpath)}, and {basename(odpath)}")
