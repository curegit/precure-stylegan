from sys import exit
from json import dump
from shutil import rmtree
from argparse import ArgumentParser
from chainer import serializers, global_config
from chainer.functions import stack
from modules.networks import Generator
from modules.argtypes import uint, natural, ufloat, positive, rate, filename, device
from modules.utilities import eprint, mkdirp, filepath, altfilepath, load_array, save_array, array2image, save_image

# Parse command line arguments
parser = ArgumentParser(allow_abbrev=False, description="Style-Based GAN's Animation Creator")
parser.add_argument("-q", "--quit", action="store_true", help="exit just before generation (debug)")
parser.add_argument("-k", "--current", action="store_true", help="save animations in current directory")
parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
parser.add_argument("-w", "--wipe", action="store_true", help="clean destination directory preliminarily")
parser.add_argument("-j", "--dump-json", action="store_true", help="save command-line arguments as a JSON file")
parser.add_argument("-o", "--no-samples", action="store_true", help="don't save sampled key images")
parser.add_argument("-i", "--image-only", action="store_true", help="don't save latent vectors")
parser.add_argument("-G", "--gif", action="store_true", help="save GIF in addition to APNG")
parser.add_argument("-W", "--webp", action="store_true", help="save WebP in addition to APNG")
parser.add_argument("-F", "--frames", action="store_true", help="save interpolated frames")
parser.add_argument("-L", "--loop", action="store_true", help="interpolate between the last and first images")
parser.add_argument("-R", "--repeat", action="store_true", help="tell the image writer explicitly that an output image should loop")
parser.add_argument("-D", "--duration", metavar="MS", type=natural, default=100, help="the display duration of each frame in milliseconds")
parser.add_argument("-N", "--interpolate", metavar="FRAME", type=uint, default=15, help="the number of frames between key images")
parser.add_argument("-P", "--prepend", metavar="FILE", nargs="+", help="add specified key images by latents to the head")
parser.add_argument("-A", "--append", metavar="FILE", nargs="+", help="add specified key images by latents to the tail")
parser.add_argument("-r", "--result", "-d", "--directory", metavar="DEST", dest="directory", default="animation", help="destination directory for generated images")
parser.add_argument("-p", "--prefix", type=filename, default="", help="filename prefix for generated images")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="growth stage, defining image resolution")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="final stage")
parser.add_argument("-c", "--channels", metavar="CH", type=natural, nargs=2, default=(512, 16), help="numbers of channels at initial stage and final stage")
parser.add_argument("-z", "--z-size", dest="size", type=natural, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", metavar="DEPTH", dest="depth", type=natural, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=uint, default=20, help="the number of images to generate")
parser.add_argument("-b", "--batch", type=natural, default=1, help="batch size, affecting memory usage")
parser.add_argument("-a", "--alpha", type=rate, default=1.0, help="the blending alpha")
parser.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="move the mean of random latent vectors to a specified one")
parser.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="the standard deviation of latent vectors")
parser.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, help="apply the truncation trick")
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
args.stage = args.maxstage if not args.stage or args.stage > args.maxstage else args.stage
h, w = generator.resolution(args.stage)
print(f"Sample Generation: {args.number}, Batch: {args.batch}")
print(f"MLP: {args.size}x{args.depth}, Stage: {args.stage}/{args.maxstage} ({w}x{h})")
print(f"Channel: {args.channels[0]} (initial) -> {args.channels[1]} (final)")
print(f"Alpha: {args.alpha}, Latent: {'Yes' if args.center is not None else 'No'}, Deviation: {args.sd}")
print(f"Truncation Trick: {args.psi if args.psi is not None else 'No'}, Device: {'CPU' if args.device < 0 else f'GPU {args.device}'}")

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

# Load additional latents
if args.prepend is not None:
	print("Loading prepended latents")
	prepend = [generator.wrap_latent(load_array(l)) for l in args.prepend]
	print(f"Prepended Latents: {len(prepend)}")
else:
	prepend = []
if args.append is not None:
	print("Loading appended latents")
	append = [generator.wrap_latent(load_array(l)) for l in args.append]
	print(f"Appended Latents: {len(append)}")
else:
	append = []

# Check the number of latents
if (len(prepend) + args.number + len(append) < 2):
	eprint("More latents required")
	exit(1)

# Init destination folder
print("Initializing destination directory")
if args.wipe:
	rmtree(args.directory, ignore_errors=True)
mkdirp(args.directory)

# Dump command-line options
if args.dump_json:
	path = filepath(args.directory, "args_quit" if args.quit else "args", "json")
	path = path if args.force else altfilepath(path)
	with open(path, mode="w", encoding="utf-8") as fp:
		dump(vars(args), fp, indent=2, sort_keys=True)

# Quit mode
if args.quit:
	print("Finished (Quit mode)")
	exit(0)

# Sampling new latents
c = 0
new_ws = []
mean_w = None if args.psi is None else generator.calculate_mean_w()
while c < args.number:
	n = min(args.number - c, args.batch)
	z = generator.generate_latent(n, center=center, sd=args.sd)
	w = generator.mapper(z)
	if args.psi is not None:
		w = generator.truncation_trick(w, psi=args.psi, mean_w=mean_w)
	new_ws += [w[i] for i in range(n)]
	if not args.no_samples:
		y = generator.generator(w, args.stage, alpha=args.alpha)
		z.to_cpu()
		y.to_cpu()
		for i in range(n):
			path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "png")
			path = path if args.force else altfilepath(path)
			save_image(y.array[i], path)
			print(f"Sample {c + i + 1}: Saved as {path}")
			if not args.image_only:
				path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "npy")
				path = path if args.force else altfilepath(path)
				save_array(z.array[i], path)
	c += n

# Prepare ws to interpolate
ws = []
for z in prepend:
	w = generator.mapper(z)
	ws.append(w if args.psi is None else generator.truncation_trick(w, psi=args.psi, mean_w=mean_w))
ws += new_ws
for z in append:
	w = generator.mapper(z)
	ws.append(w if args.psi is None else generator.truncation_trick(w, psi=args.psi, mean_w=mean_w))

# Generator of interpolated ws
def interpolate_iterator(ws, middles=15, loop=True, closed=True):
	n = len(ws)
	w_pairs = [(ws[i], ws[(i + 1) % n]) for i in range(n if loop else n - 1)]
	for w1, w2 in w_pairs:
		yield w1
		for i in range(1, middles + 1):
			yield w1 + (i / (middles + 1)) * (w2 - w1)
	if closed and not loop:
		yield ws[n - 1]

# Batched ws generator
def batch_iterator(iterator, n):
	ws = []
	for w in iterator:
		ws.append(w)
		if len(ws) == n:
			yield stack(ws)
			ws = []
	if len(ws) > 0:
		yield stack(ws)

# Generate frames
c = 0
images = []
w_iterator = batch_iterator(interpolate_iterator(ws, args.interpolate, args.loop), args.batch)
for i, w in enumerate(w_iterator, 1):
	n = w.shape[0]
	y = generator.generator(w, args.stage, alpha=args.alpha)
	y.to_cpu()
	for i in range(n):
		image = array2image(y.array[i])
		images.append(image)
		print(f"Frame {c + i + 1}: Finished")
		if args.frames:
			path = filepath(args.directory, f"{args.prefix}f{c + i + 1}", "png")
			path = path if args.force else altfilepath(path)
			image.save(path)
			print(f"Frame {c + i + 1}: Saved as {path}")
	c += n

# Merge frames
for ext in ["png"] + (["webp"] if args.webp else []) + (["gif"] if args.gif else []):
	path = filepath("." if args.current else args.directory, f"{args.prefix}analogy", ext)
	path = path if args.force else altfilepath(path)
	if args.repeat:
		images[0].save(path, save_all=True, duration=args.duration, append_images=images[1:], loop=0)
	else:
		images[0].save(path, save_all=True, duration=args.duration, append_images=images[1:])
