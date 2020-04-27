from shutil import rmtree
from argparse import ArgumentParser
from chainer import serializers, global_config
from chainer.functions import stack
from modules.networks import Generator
from modules.argtypes import uint, natural, ufloat, positive, rate, filename, device
from modules.utilities import mkdirp, filepath, altfilepath, load_array, save_array, array2image, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Generator")
parser.add_argument("-q", "--quit", action="store_true", help="")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-w", "--wipe", action="store_true", help="")
parser.add_argument("-i", "--image-only", action="store_true", help="")

#parser.add_argument("-o", "--", action="store_true", help="")
parser.add_argument("-F", "--frames", action="store_true", help="")
parser.add_argument("-L", "--loop", action="store_true", help="")
parser.add_argument("-R", "--repeat", action="store_true", help="")
parser.add_argument("-D", "--duration", type=natural, default=50, help="")
parser.add_argument("-N", "--interpolate", type=uint, default=63, help="")
parser.add_argument("-P", "--prepend", metavar="FILE", nargs="*", default=[], help="")
parser.add_argument("-A", "--append", metavar="FILE", nargs="*", default=[], help="")

parser.add_argument("-r", "--result", "-d", "--directory", metavar="DEST", dest="directory", default="animation", help="destination directory for generated images")
parser.add_argument("-p", "--prefix", type=filename, default="", help="filename prefix for generated images")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="growth stage, defining image resolution")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="final stage")
parser.add_argument("-c", "--channels", metavar="CH", type=natural, nargs=2, default=(512, 16), help="numbers of channels at initial stage and final stage")
parser.add_argument("-z", "--z-size", dest="size", type=natural, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", metavar="DEPTH", dest="depth", type=natural, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=uint, default=1, help="the number of images to generate")
parser.add_argument("-b", "--batch", type=natural, default=1, help="batch size, affecting memory usage")
parser.add_argument("-a", "--alpha", type=rate, default=1.0, help="")
parser.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="")
parser.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="")
parser.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, help="")
parser.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=-1, help="use specified GPU or CPU device")
args = parser.parse_args()

# Config chainer
global_config.train = False
global_config.autotune = True
global_config.cudnn_deterministic = True

# Init model
print("Initializing model")
generator = Generator(args.size, args.depth, args.channels, args.maxstage)

# Print information

# Load model
if args.generator is not None:
	print("Loading generator")
	serializers.load_hdf5(args.generator, generator)

# GPU setting
if args.device >= 0:
	print("Converting to GPU")
	generator.to_gpu(args.device)

# Load center latent
if args.center is not None:
	print("Loading latent")
	center = generator.wrap_latent(load_array(args.center))
else:
	center = None

# Load
prepend = [generator.wrap_latent(load_array(l)) for l in args.prepend]
append = [generator.wrap_latent(load_array(l)) for l in args.append]

# Init destination folder
print("Initializing destination directory")
if args.wipe:
	rmtree(args.directory, ignore_errors=True)
mkdirp(args.directory)

# Quit mode
if args.quit:
	print("Finished (Quit mode)")
	exit(0)

# Generate images
c = 0
new_ws = []
mean_w = None if args.psi is None else generator.calculate_mean_w()
while c < args.number:
	n = min(args.number - c, args.batch)
	z = generator.generate_latent(n, center=center, sd=args.sd)
	w = generator.mapper(z)
	if args.psi is not None:
		w = generator.truncation_trick(w, psi=args.psi, mean_w=mean_w)
	if True:
		y = generator.generator(w, args.stage, alpha=args.alpha)
		y.to_cpu()
	z.to_cpu()
	for i in range(n):
		new_ws.append(w[i])
		if True:
			path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "png")
			path = path if args.force else altfilepath(path)
			save_image(y.array[i], path)
			#print(f"{c + i + 1}/{args.number}: Saved as {path}")
		if not args.image_only:
			path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "npy")
			path = path if args.force else altfilepath(path)
			save_array(z.array[i], path)
	c += n

#
ws = []
for z in prepend:
	w = generator.mapper(z)
	ws.append(w if args.psi is None else generator.truncation_trick(w, psi=args.psi, mean_w=mean_w))
ws += new_ws
for z in append:
	w = generator.mapper(z)
	ws.append(w if args.psi is None else generator.truncation_trick(w, psi=args.psi, mean_w=mean_w))

#
def interpolate_iterator(ws, middles=15, loop=True, closed=True):
	n = len(ws)
	w_pairs = [(ws[i], ws[(i + 1) % n]) for i in range(n if loop else n - 1)]
	for w1, w2 in w_pairs:
		yield w1
		for i in range(1, middles + 1):
			yield w1 + (i / (middles + 1)) * (w2 - w1)
	if closed and not loop:
		yield ws[n - 1]

# Batched w generator
def batch_iterator(iterator, n):
	ws = []
	for w in iterator:
		ws.append(w)
		if len(ws) == n:
			yield stack(ws)
			ws = []
	if len(ws) > 0:
		yield stack(ws)

# Generate images
c = 0
images = []
w_iterator = batch_iterator(interpolate_iterator(ws, args.interpolate, args.loop), args.batch)
for i, w in enumerate(w_iterator, 1):
	print(w.shape)
	n = w.shape[0]
	y = generator.generator(w, args.stage, alpha=args.alpha)
	y.to_cpu()
	for i in range(n):
		image = array2image(y.array[i])
		images.append(image)
		if args.frames:
			path = filepath(args.directory, f"{args.prefix}f{c + i + 1}", "png")
			path = path if args.force else altfilepath(path)
			image.save(path)
			#print(f"{c + i + 1}/{args.number}: Saved as {path}")
	c += n

# Merge frames
#ext =
#path =
images[0].save('analogy.gif', save_all=True, duration=args.duration, append_images=images[1:], loop=0)
images[0].save('analogy.webp', save_all=True, duration=args.duration, append_images=images[1:], loop=args.repeat)
images[0].save('analogy.png', save_all=True, duration=args.duration, append_images=images[1:], loop=args.repeat)
