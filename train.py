from random import random
from os.path import basename
from shutil import rmtree
from datetime import datetime
from argparse import ArgumentParser
from chainer import optimizers, serializers, global_config
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer, extensions, make_extension
from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.networks import Generator, Discriminator
from modules.argtypes import uint, natural, ufloat, positive, rate, device
from modules.utilities import eprint, mkdirp, filepath, altfilepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("dataset", metavar="DATASET_DIR", help="dataset directory which stores images")
parser.add_argument("-q", "--quit", action="store_true", help="")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
parser.add_argument("-k", "--current", action="store_true", help="save completed models in current directory")
parser.add_argument("-j", "--no-netinfo", dest="noinfo", action="store_true", help="")
parser.add_argument("-y", "--no-datetime", dest="nodate", action="store_true", help="don't add date time prefix to completed model files")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-w", "--wipe", action="store_true", help="")
parser.add_argument("-r", "--result", "--directory", metavar="DEST", dest="result", default="results", help="destination directory for models, logs, middle images, and so on")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained generator to load and retrain")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="HDF5 file of serialized trained discriminator to load and retrain")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="snapshot of optimizers of mapper, generator, and discriminator")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="growth stage to train")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="")
parser.add_argument("-c", "--channels", metavar="CH", type=natural, nargs=2, default=(512, 16), help="")
parser.add_argument("-z", "--z-size", dest="size", type=natural, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="depth", type=natural, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=uint, default=10, help="the number of middle images to generate")
parser.add_argument("-b", "--batch", type=natural, default=4, help="batch size, affecting memory usage")
parser.add_argument("-e", "--epoch", type=natural, default=1, help="")
parser.add_argument("-a", "--alpha", type=rate, default=0.0, help="")
parser.add_argument("-t", "--delta", type=positive, default=0.00005, help="")
parser.add_argument("-G", "--gamma", "--l2-batch", dest="gamma", type=ufloat, default=10, help="")
parser.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="")
parser.add_argument("-i", "--style-mixing", metavar="RATE", dest="mix", type=rate, default=0.5, help="")
parser.add_argument("-u", "--print-interval", metavar="ITER", dest="print", type=uint, nargs=2, default=(5, 500), help="")
parser.add_argument("-l", "--write-interval", metavar="ITER", dest="write", type=uint, nargs=4, default=(1000, 3000, 500, 500), help="")
parser.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Config chainer
global_config.train = True
global_config.autotune = True
global_config.cudnn_deterministic = False

# Init models
print("Initializing models")
generator = Generator(args.size, args.depth, args.channels, args.maxstage)
discriminator = Discriminator(args.channels, args.maxstage)

# Prepare dataset
print("Loading dataset" if args.preload else "Scanning dataset")
h, w = generator.resolution(args.stage)
dataset = StyleGanDataset(args.dataset, (w, h), args.preload)
iterator = MultiprocessIterator(dataset, batch_size=args.batch, repeat=True, shuffle=True, n_prefetch=4)
n = dataset.length()
if n < 1:
	eprint("No image found in dataset directory")
	exit(1)

# Print information
print(f"MLP: {args.size}x{args.depth}, Stage: {args.stage}/{args.maxstage} ({w}x{h})")
print(f"Channel: {args.channels[0]} (initial) -> {args.channels[1]} (final)")
print(f"Epoch: {args.epoch}, Batch: {args.batch}, Dataset Images: {n}")
print(f"Mixing Rate: {args.mix * 100:.1f}%, Initial Alpha: {args.alpha:.3f}, Delta: {args.delta} (/iter)")
print(f"Objective: {'Least Squares' if args.lsgan else 'Adversarial'}, Gamma: {args.gamma}, Device: {'CPU' if args.device < 0 else f'GPU {args.device}'}")

# Load models
if args.generator is not None:
	print("Loading generator")
	serializers.load_hdf5(args.generator, generator)
if args.discriminator is not None:
	print("Loading discriminator")
	serializers.load_hdf5(args.discriminator, discriminator)

# GPU setting
if args.device >= 0:
	print("Converting to GPU")
	generator.to_gpu(args.device)
	discriminator.to_gpu(args.device)

# Init optimizers
print("Initializing optimizers")
mapper_optimizer = optimizers.Adam(alpha=0.00001, beta1=0.0, beta2=0.99, eps=1e-08).setup(generator.mapper)
generator_optimizer = optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99, eps=1e-08).setup(generator.generator)
discriminator_optimizer = optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99, eps=1e-08).setup(discriminator)

# Load optimizers
if args.optimizers is not None:
	print("Loading mapper's optimizer")
	serializers.load_hdf5(args.optimizers[0], mapper_optimizer)
	print("Loading generator's optimizer")
	serializers.load_hdf5(args.optimizers[1], generator_optimizer)
	print("Loading discriminator's optimizer")
	serializers.load_hdf5(args.optimizers[2], discriminator_optimizer)

# Prepare updater
updater = StyleGanUpdater(generator, discriminator, iterator, {"mapper": mapper_optimizer, "generator": generator_optimizer, "discriminator": discriminator_optimizer}, args.device, args.stage, args.mix, args.alpha, args.delta, args.gamma, args.lsgan)

# Init result directory
print("Initializing destination directory")
if args.wipe:
	rmtree(args.result, ignore_errors=True)
mkdirp(args.result)

# Define extension to output images in progress
def save_middle_images(generator, stage, directory, number, batch, mix, force=True):
	@make_extension()
	def func(trainer):
		c = 0
		mixing = mix > random()
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			mix_z = generator.generate_latent(n) if mixing else None
			y = generator(z, stage, trainer.updater.alpha, mix_z)
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "png")
				path = path if force else altfilepath(path)
				save_image(y.array[i], path)
			c += n
	return func

# Define extension to save models in progress
def save_middle_models(generator, discriminator, stage, directory, device, force=True):
	@make_extension()
	def func(trainer):
		generator.to_cpu()
		discriminator.to_cpu()
		path = filepath(directory, f"gen_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, generator)
		path = filepath(directory, f"dis_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, discriminator)
		if device >= 0:
			generator.to_gpu(device)
			discriminator.to_gpu(device)
	return func

# Define extension to save optimizers in progress
def save_middle_optimizers(mapper_optimizer, generator_optimizer, discriminator_optimizer, stage, directory, force=True):
	@make_extension()
	def func(trainer):
		path = filepath(directory, f"mopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, mapper_optimizer)
		path = filepath(directory, f"gopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, generator_optimizer)
		path = filepath(directory, f"dopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, discriminator_optimizer)
	return func

# Prepare trainer
logpath = filepath(args.result, "report", "log")
logname = basename(logpath if args.force else altfilepath(logpath))
plotpath = filepath(args.result, "report", "png")
plotname = basename(plotpath if args.force else altfilepath(plotpath))
trainer = Trainer(updater, (args.epoch, "epoch"), out=args.result)
if args.print[0] > 0: trainer.extend(extensions.ProgressBar(update_interval=args.print[0]))
if args.print[1] > 0: trainer.extend(extensions.PrintReport(["epoch", "iteration", "alpha", "loss (gen)", "loss (dis)", "loss (grad)"], extensions.LogReport(trigger=(args.print[1], "iteration"))))
if args.write[0] > 0: trainer.extend(save_middle_images(generator, args.stage, args.result, args.number, args.batch, args.mix, args.force), trigger=(args.write[0], "iteration"))
if args.write[1] > 0: trainer.extend(save_middle_models(generator, discriminator, args.stage, args.result, args.device, args.force), trigger=(args.write[1], "iteration"))
if args.write[1] > 0: trainer.extend(save_middle_optimizers(mapper_optimizer, generator_optimizer, discriminator_optimizer, args.stage, args.result, args.force), trigger=(args.write[1], "iteration"))
if args.write[2] > 0: trainer.extend(extensions.LogReport(trigger=(args.write[2], "iteration"), filename=logname))
if args.write[3] > 0: trainer.extend(extensions.PlotReport(["alpha", "loss (gen)", "loss (dis)", "loss (grad)"], "iteration", trigger=(args.write[3], "iteration"), filename=plotname))

# Quit mode
if args.quit:
	print("Finished (Quit mode)")
	exit(0)

# Run ML
trainer.run()

# Save models
print("Saving models")
generator.to_cpu()
discriminator.to_cpu()
n = f"s{args.stage}x{args.maxstage}c{args.channels[0]}-{args.channels[1]}z{args.size}m{args.depth}"
t = datetime.now().strftime("%m%d%H")
gname = f"gen{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
dname = f"dis{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
gpath = filepath("." if args.current else args.result, gname, "hdf5")
dpath = filepath("." if args.current else args.result, dname, "hdf5")
gpath = gpath if args.force else altfilepath(gpath)
dpath = dpath if args.force else altfilepath(dpath)
serializers.save_hdf5(gpath, generator)
print(f"Generator: saved as {basename(gpath)}")
serializers.save_hdf5(dpath, discriminator)
print(f"Discriminator: saved as {basename(dpath)}")

# Save optimizers
print("Saving optimizers")
omname = f"mopt{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
ogname = f"gopt{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
odname = f"dopt{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
ompath = filepath("." if args.current else args.result, omname, "hdf5")
ogpath = filepath("." if args.current else args.result, ogname, "hdf5")
odpath = filepath("." if args.current else args.result, odname, "hdf5")
ompath = ompath if args.force else altfilepath(ompath)
ogpath = ogpath if args.force else altfilepath(ogpath)
odpath = odpath if args.force else altfilepath(odpath)
serializers.save_hdf5(ompath, mapper_optimizer)
print(f"Mapper's optimizer: saved as {basename(ompath)}")
serializers.save_hdf5(ogpath, generator_optimizer)
print(f"Generator's optimizer: saved as {basename(ogpath)}")
serializers.save_hdf5(odpath, discriminator_optimizer)
print(f"Discriminator's optimizer: saved as {basename(odpath)}")
