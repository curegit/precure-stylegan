from os.path import basename
from shutil import rmtree
from argparse import ArgumentParser
from chainer import serializers
from modules.networks import Generator
from modules.utilities import mkdirp, filepath, altfilepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Generator")
parser.add_argument("-q", "--quit", action="store_true", help="")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-w", "--wipe", action="store_true", help="")
parser.add_argument("-r", "--result", "-d", "--directory", metavar="DEST", dest="directory", default="images", help="destination directory for generated images")
parser.add_argument("-p", "--prefix", help="filename prefix for generated images")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="growth stage, defining image resolution")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="final stage")
parser.add_argument("-c", "--channels", metavar="CH", type=int, nargs=2, default=(512, 16), help="numbers of channels at initial stage and final stage")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", metavar="DEPTH", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=int, default=1, help="the number of images to generate")
parser.add_argument("-b", "--batch", type=int, default=1, help="batch size, affecting memory usage")
parser.add_argument("-a", "--alpha", type=float, default=1.0, help="")
parser.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=float, help="")
parser.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=int, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Validate arguments
number = max(0, args.number)
batch = max(1, args.batch)
alpha = max(0.0, min(1.0, args.alpha))
stage = min(args.stage, args.maxstage)
channels = (max(1, args.channels[0]), max(1, args.channels[1]))
size = max(1, args.size)
depth = max(1, args.mlp)
device = max(-1, args.device)
prefix = basename(args.prefix or "")

# Init model
print("Initializing model")
generator = Generator(size, depth, channels, args.maxstage)

# Print information
h, w = generator.resolution(stage)
print(f"Total Generation: {number}, Batch: {batch}")
print(f"MLP: {size}x{depth}, Stage: {stage}/{args.maxstage} ({w}x{h})")
print(f"Channel: {channels[0]} (initial) -> {channels[1]} (final)")
print(f"Device: {'CPU' if device < 0 else f'GPU {device}'}")

# Init destination folder
print("Initializing destination directory")
if args.wipe:
	rmtree(args.directory, ignore_errors=True)
mkdirp(args.directory)

# Load model
if args.generator is not None:
	print("Loading generator")
	serializers.load_hdf5(args.generator, generator)

# GPU setting
if device >= 0:
	print("Converting to GPU")
	generator.to_gpu(device)

# Quit mode
if args.quit:
	print("Finished (Quit mode)")
	exit(0)

# Generate images
c = 0
while c < number:
	n = min(number - c, batch)
	z = generator.generate_latent(n)
	y = generator(z, stage, alpha=alpha, psi=args.psi)
	y.to_cpu()
	for i in range(n):
		path = filepath(args.directory, f"{prefix}{c + i + 1}", "png")
		path = path if args.force else altfilepath(path)
		save_image(y.array[i], path)
		print(f"{c + i + 1}/{number}: Saved as {basename(path)}")
	c += n
