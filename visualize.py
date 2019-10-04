import numpy as np
import pydot as dot
import chainer.computational_graph as cg
from modules.networks import Generator

# Model params
stage = 9
z_size = 512

# Make generator graph
gen = Generator(z_size)
z = np.array(range(z_size), dtype=np.float32).reshape(1, z_size)
y = gen(z, stage)
d = cg.build_computational_graph([y])
with open("generator.dot", "w") as o:
	o.write(d.dump())

# Save as PDF
g = dot.graph_from_dot_file("generator.dot")[0]
g.write_pdf("generator.pdf")
