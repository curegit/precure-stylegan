from random import random
from os.path import basename
from json import dump
from shutil import rmtree
from datetime import datetime
from argparse import ArgumentParser
from chainer import optimizers, serializers, global_config
from chainer.iterators import SerialIterator
from chainer.training import Trainer, extensions, make_extension
from modules.updater import StyleGanUpdater
from modules.dataset import StyleGanDataset
from modules.networks import Generator, Discriminator
from modules.argtypes import uint, natural, ufloat, positive, rate, device
from modules.utilities import eprint, mkdirp, filepath, altfilepath, save_array, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Trainer")
parser.add_argument("dataset", metavar="DATASET_DIR", help="dataset directory which stores images")
parser.add_argument("-q", "--quit", action="store_true", help="exit just before training (debug)")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
parser.add_argument("-k", "--current", action="store_true", help="save completed models in current directory")
parser.add_argument("-j", "--no-netinfo", dest="noinfo", action="store_true", help="don't add architecture info to completed model file names")
parser.add_argument("-y", "--no-datetime", dest="nodate", action="store_true", help="don't add date time prefix to completed model file names")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-w", "--wipe", action="store_true", help="clean destination directory")
parser.add_argument("-r", "--result", "--directory", metavar="DEST", dest="result", default="results", help="destination directory for models, logs, middle images, and so on")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained generator to load and retrain")
parser.add_argument("-G", "--averaged-generator", metavar="FILE", dest="averaged", help="HDF5 file of serialized averaged generator to load and continue")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="HDF5 file of serialized trained discriminator to load and retrain")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="snapshot of optimizers of mapper, generator, and discriminator")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=1, help="growth stage to train")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="the number of stages")
parser.add_argument("-c", "--channels", metavar="CH", type=natural, nargs=2, default=(512, 16), help="numbers of channels at initial stage and final stage")
parser.add_argument("-z", "--z-size", dest="size", type=natural, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="depth", type=natural, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=uint, default=10, help="the number of middle images to generate")
parser.add_argument("-b", "--batch", type=natural, default=16, help="batch size, affecting memory usage")
parser.add_argument("-e", "--epoch", type=natural, default=1, help="training duration in epoch")
parser.add_argument("-a", "--alpha", type=rate, default=0.0, help="the blending alpha")
parser.add_argument("-t", "--delta", type=positive, default=0.00005, help="increasing amount of the blending alpha per iter")
parser.add_argument("-R", "--gamma", "--r1-gamma", dest="gamma", type=ufloat, default=10, help="coefficient of R1 regularization (set 0 to disable)")
parser.add_argument("-D", "--decay", type=rate, default=0.999, help="")
parser.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="use the least squares loss function instead of the logistic loss function")
parser.add_argument("-i", "--style-mixing", metavar="RATE", dest="mix", type=rate, default=0.5, help="application rate of the mixing regularization")
parser.add_argument("-S", "--sgd", metavar="LR", type=positive, nargs=3, help="use SGD optimizers")
parser.add_argument("-A", "--adam-alphas", metavar="ALPHA", type=positive, nargs=3, default=(0.00001, 0.001, 0.001), help="Adam's coefficients of learning rates of mapper, generator, and discriminator")
parser.add_argument("-B", "--adam-betas", metavar=("BETA1", "BETA2"), type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
parser.add_argument("-u", "--print-interval", metavar="ITER", dest="print", type=uint, nargs=2, default=(5, 500), help="")
parser.add_argument("-l", "--write-interval", metavar="ITER", dest="write", type=uint, nargs=4, default=(1000, 3000, 500, 500), help="")
parser.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Config chainer
global_config.train = True
global_config.autotune = True
global_config.cudnn_deterministic = False

# Init models
print("Initializing models...")
generator = Generator(args.size, args.depth, args.channels, args.maxstage)
discriminator = Discriminator(args.channels, args.maxstage)

# Prepare dataset
print("Loading dataset..." if args.preload else "Scanning dataset...")
args.stage = args.maxstage if args.stage > args.maxstage else args.stage
h, w = generator.resolution(args.stage)
dataset = StyleGanDataset(args.dataset, (w, h), args.preload)
n = dataset.length()
if n < 1:
	eprint("No image found in dataset directory")
	exit(1)
iterator = SerialIterator(dataset, batch_size=args.batch, repeat=True, shuffle=True)

# Print information
print(f"MLP: {args.size}x{args.depth}, Stage: {args.stage}/{args.maxstage} ({w}x{h})")
print(f"Channel: {args.channels[0]} (initial) -> {args.channels[1]} (final)")
print(f"Epoch: {args.epoch}, Batch: {args.batch}, EMA Decay: {args.decay}, Dataset Images: {n}")
print(f"Mixing Rate: {args.mix * 100}%, Initial Alpha: {args.alpha}, Delta: {args.delta} (/iter)")
print(f"Objective: {'Least Squares' if args.lsgan else 'Adversarial'}, Gamma: {args.gamma}, Device: {'CPU' if args.device < 0 else f'GPU {args.device}'}")

# Load models
if args.generator is not None:
	print("Loading generator...")
	serializers.load_hdf5(args.generator, generator)
if args.discriminator is not None:
	print("Loading discriminator...")
	serializers.load_hdf5(args.discriminator, discriminator)

#
averaged_generator = generator.copy("copy")
if args.averaged is not None:
	print("Loading averaged generator...")
	serializers.load_hdf5(args.averaged, averaged_generator)

# GPU setting
if args.device >= 0:
	print("Converting to GPU...")
	generator.to_gpu(args.device)
	averaged_generator.to_gpu(args.device)
	discriminator.to_gpu(args.device)

# Init optimizers
print("Initializing optimizers...")
if args.sgd is None:
	mapper_optimizer = optimizers.Adam(alpha=args.adam_alphas[0], beta1=args.adam_betas[0], beta2=args.adam_betas[1], eps=1e-08).setup(generator.mapper)
	print(f"Mapper: Adam(alpha: {args.adam_alphas[0]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
	generator_optimizer = optimizers.Adam(alpha=args.adam_alphas[1], beta1=args.adam_betas[0], beta2=args.adam_betas[1], eps=1e-08).setup(generator.generator)
	print(f"Generator: Adam(alpha: {args.adam_alphas[1]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
	discriminator_optimizer = optimizers.Adam(alpha=args.adam_alphas[2], beta1=args.adam_betas[0], beta2=args.adam_betas[1], eps=1e-08).setup(discriminator)
	print(f"Discriminator: Adam(alpha: {args.adam_alphas[2]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
else:
	mapper_optimizer = optimizers.SGD(args.sgd[0]).setup(generator.mapper)
	print(f"Mapper: SGD(learning rate: {args.sgd[0]})")
	generator_optimizer = optimizers.SGD(args.sgd[1]).setup(generator.generator)
	print(f"Generator: SGD(learning rate: {args.sgd[1]})")
	discriminator_optimizer = optimizers.SGD(args.sgd[2]).setup(discriminator)
	print(f"Discriminator: SGD(learning rate: {args.sgd[2]})")

# Load optimizers
if args.optimizers is not None:
	print("Loading mapper's optimizer...")
	serializers.load_hdf5(args.optimizers[0], mapper_optimizer)
	print("Loading generator's optimizer...")
	serializers.load_hdf5(args.optimizers[1], generator_optimizer)
	print("Loading discriminator's optimizer...")
	serializers.load_hdf5(args.optimizers[2], discriminator_optimizer)

# Prepare updater
updater = StyleGanUpdater(generator, averaged_generator, discriminator, iterator, {"mapper": mapper_optimizer, "generator": generator_optimizer, "discriminator": discriminator_optimizer}, args.device, args.stage, args.mix, args.alpha, args.delta, args.gamma, args.decay, args.lsgan)

# Init result directory
print("Initializing destination directory...")
if args.wipe:
	rmtree(args.result, ignore_errors=True)
mkdirp(args.result)

# Dump command-line options
path = filepath(args.result, "args_quit" if args.quit else "args", "json")
path = path if args.force else altfilepath(path)
with open(path, mode="w", encoding="utf-8") as fp:
	dump(vars(args), fp, indent=2, sort_keys=True)

# Define extension to output images in progress
def save_middle_images(generator, stage, directory, number, batch, mix, force=True, save_latent=True):
	@make_extension()
	def func(trainer):
		c = 0
		mixing = mix > random()
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			mix_z = generator.generate_latent(n) if mixing else None
			y = generator(z, stage, trainer.updater.alpha, mix_z)
			z.to_cpu()
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "png")
				path = path if force else altfilepath(path)
				save_image(y.array[i], path)
				if save_latent:
					path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "npy")
					path = path if force else altfilepath(path)
					save_array(z.array[i], path)
			c += n
	return func

# Define extension to save models in progress
def save_middle_models(generator, averaged_generator, discriminator, stage, directory, device, force=True):
	@make_extension()
	def func(trainer):
		generator.to_cpu()
		averaged_generator.to_cpu()
		discriminator.to_cpu()
		path = filepath(directory, f"gen_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, generator)
		path = filepath(directory, f"avgen_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, averaged_generator)
		path = filepath(directory, f"dis_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, discriminator)
		if device >= 0:
			generator.to_gpu(device)
			averaged_generator.to_gpu(device)
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
plotpath = filepath(args.result, "plot", "png")
plotname = basename(plotpath if args.force else altfilepath(plotpath))
trainer = Trainer(updater, (args.epoch, "epoch"), out=args.result)
if args.print[0] > 0: trainer.extend(extensions.ProgressBar(update_interval=args.print[0]))
if args.print[1] > 0: trainer.extend(extensions.PrintReport(["epoch", "iteration", "alpha", "loss (gen)", "loss (dis)", "loss (grad)"], extensions.LogReport(trigger=(args.print[1], "iteration"), log_name=None)))
if args.write[0] > 0: trainer.extend(save_middle_images(averaged_generator, args.stage, args.result, args.number, args.batch, args.mix, args.force), trigger=(args.write[0], "iteration"))
if args.write[1] > 0: trainer.extend(save_middle_models(generator, averaged_generator, discriminator, args.stage, args.result, args.device, args.force), trigger=(args.write[1], "iteration"))
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
print("Saving models...")
generator.to_cpu()
averaged_generator.to_cpu()
discriminator.to_cpu()
n = f"s{args.stage}x{args.maxstage}c{args.channels[0]}-{args.channels[1]}z{args.size}m{args.depth}"
t = datetime.now().strftime("%m%d%H")
gname = f"gen{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
avname = f"avgen{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
dname = f"dis{'' if args.noinfo else f'_{n}'}{'' if args.nodate else f'_{t}'}"
gpath = filepath("." if args.current else args.result, gname, "hdf5")
avpath = filepath("." if args.current else args.result, avname, "hdf5")
dpath = filepath("." if args.current else args.result, dname, "hdf5")
gpath = gpath if args.force else altfilepath(gpath)
avpath = avpath if args.force else altfilepath(avpath)
dpath = dpath if args.force else altfilepath(dpath)
serializers.save_hdf5(gpath, generator)
print(f"Generator: saved as {gpath}")
serializers.save_hdf5(avpath, averaged_generator)
print(f"Averaged Generator: saved as {avpath}")
serializers.save_hdf5(dpath, discriminator)
print(f"Discriminator: saved as {dpath}")

# Save optimizers
print("Saving optimizers...")
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
print(f"Mapper's optimizer: saved as {ompath}")
serializers.save_hdf5(ogpath, generator_optimizer)
print(f"Generator's optimizer: saved as {ogpath}")
serializers.save_hdf5(odpath, discriminator_optimizer)
print(f"Discriminator's optimizer: saved as {odpath}")
