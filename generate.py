from os.path import basename
from argparse import ArgumentParser
from chainer import serializers
from modules.networks import Generator
from modules.utilities import mkdirp, filepath, altfilepath, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Generator")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-d", "--directory", metavar="DEST", default=".", help="destination directory for generated images")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized train model to load")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="growth stage, defining image resolution")
parser.add_argument("-z", "--z-size", dest="size", type=int, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", dest="mlp", type=int, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=int, default=1, help="the number of images to generate")
parser.add_argument("-b", "--batch", type=int, default=1, help="batch size, affecting memory usage")
parser.add_argument("-v", "--device", type=int, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Validate arguments
number = max(0, args.number)
batch = max(1, args.batch)
size = max(1, args.size)
depth = max(1, args.mlp)
device = max(-1, args.device)

# Init model
print("Initializing model")
generator = Generator(size, depth)

# Print information
h, w = generator.resolution(args.stage)
print(f"Total: {number}, Batch: {batch}")
print(f"MLP: {size}x{depth}, Stage: {args.stage} ({w}x{h})")
print(f"Device: {'CPU' if device < 0 else f'GPU {device}'}")

# Make destination folder
mkdirp(args.directory)

# Load model
if args.generator is not None:
	print("Loading generator")
	serializers.load_hdf5(args.generator, generator)

if device >= 0:
	print("Converting to GPU")
	generator.to_gpu(device)

# Generate images
c = 0
while c < number:
	n = min(number - c, batch)
	z = generator.generate_latent(n)
	y = generator(z, args.stage)
	y.to_cpu()
	for i in range(n):
		path = filepath(args.directory, f"{c + i + 1}", "png")
		path = path if args.force else altfilepath(path)
		save_image(y.array[i], path)
		print(f"{c + i + 1}/{number}: Saved as {basename(path)}")
	c += n
