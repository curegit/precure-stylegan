from pydot import graph_from_dot_data
from chainer import Variable
from chainer.computational_graph import build_computational_graph
from modules.networks import Generator, Discriminator
from modules.utilities import mkdirp, filepath, filerelpath

# Model parameters
stage = 4
max_stage = 7
channels = (512, 16)
z_size = 512
depth = 8

# Evaluation parameters
batch = 4
alpha = 0.5
mix_stage = 2
psi = 0.7

# Graph style configuration
gvarstyle = {"fillcolor": "#5edbf1", "shape": "record", "style": "filled"}
gfuncstyle = {"fillcolor": "#ffa9e0", "shape": "record", "style": "filled"}
dvarstyle = {"fillcolor": "#7a9fe6", "shape": "record", "style": "filled"}
dfuncstyle = {"fillcolor": "#fea21d", "shape": "record", "style": "filled"}

# Output settings
directory = "graphs"
filename_g = "generator"
filename_d = "discriminator"

# Make the destination directory
path = filerelpath(directory)
mkdirp(path)

# Build the generator's graph
gen = Generator(z_size, depth, channels, max_stage)
z = gen.generate_latent(batch)
mix = gen.generate_latent(batch)
mean = gen.calculate_mean_w()
i = gen(z, stage, alpha, mix, mix_stage, psi, mean)
dg = build_computational_graph([i], variable_style=gvarstyle, function_style=gfuncstyle).dump()

# Build the discriminator's graph
dis = Discriminator(channels, max_stage)
y = dis(Variable(i.array), stage, alpha)
dd = build_computational_graph([y], variable_style=dvarstyle, function_style=dfuncstyle).dump()

# Save the graphs as dot files
dg_path = filepath(path, filename_g, "dot")
dd_path = filepath(path, filename_d, "dot")
with open(dg_path, "w") as f:
	f.write(dg)
print(f"Saved: {dg_path}")
with open(dd_path, "w") as f:
	f.write(dd)
print(f"Saved: {dd_path}")

# Save the graphs as PDFs
pg_path = filepath(path, filename_g, "pdf")
pd_path = filepath(path, filename_d, "pdf")
gg = graph_from_dot_data(dg)[0]
gd = graph_from_dot_data(dd)[0]
gg.write_pdf(pg_path)
print(f"Saved: {pg_path}")
gd.write_pdf(pd_path)
print(f"Saved: {pd_path}")
