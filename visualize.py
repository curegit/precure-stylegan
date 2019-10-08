import pydot as dot
import chainer.computational_graph as cg
from modules.networks import Generator, Discriminator
from modules.utilities import mkdirp, filepath, filerelpath

# Model params
stage = 3
z_size = 128

# Evaluation config
batch = 2
alpha = 0.5

# Graph style config
gvarstyle = {"fillcolor": "#5edbf1", "shape": "record", "style": "filled"}
gfuncstyle = {"fillcolor": "#ffa9e0", "shape": "record", "style": "filled"}
dvarstyle = {"fillcolor": "#7a9fe6", "shape": "record", "style": "filled"}
dfuncstyle = {"fillcolor": "#fea21d", "shape": "record", "style": "filled"}

# Destination config
directory = "graphs"
filename_g = "generator"
filename_d = "discriminator"

# Make directory
path = filerelpath(directory)
mkdirp(path)

# Make generator graph
gen = Generator(z_size)
z = gen.generate_latent(batch)
i = gen(z, stage, alpha)
dg = cg.build_computational_graph([i], variable_style=gvarstyle, function_style=gfuncstyle).dump()

# Make Discriminator graph
dis = Discriminator()
y = dis(i.array, stage, alpha)
dd = cg.build_computational_graph([y], variable_style=dvarstyle, function_style=dfuncstyle).dump()

# Save as dot file
dg_path = filepath(path, filename_g, "dot")
dd_path = filepath(path, filename_d, "dot")
with open(dg_path, "w") as f:
	f.write(dg)
with open(dd_path, "w") as f:
	f.write(dd)

# Save as PDF
pg_path = filepath(path, filename_g, "pdf")
pd_path = filepath(path, filename_d, "pdf")
gg = dot.graph_from_dot_data(dg)[0]
gd = dot.graph_from_dot_data(dd)[0]
gg.write_pdf(pg_path)
gd.write_pdf(pd_path)
