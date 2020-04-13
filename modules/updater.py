from random import random
from chainer import grad
from chainer.reporter import report
from chainer.training import StandardUpdater
from chainer.functions import sum, batch_l2_norm_squared, softplus

# Updater for StyleGAN
class StyleGanUpdater(StandardUpdater):

	def __init__(self, generator, discriminator, iterator, optimizer, device, stage, mixing=0.5, alpha=0.0, delta=0.00005, gamma=10, lsgan=False):
		super().__init__(iterator, optimizer, device=device)
		self.alpha = alpha
		self.delta = delta
		self.gamma = gamma
		self.lsgan = lsgan
		self.stage = stage
		self.mixing = mixing
		self.generator = generator
		self.discriminator = discriminator
		self.mapper_optimizer = optimizer["mapper"]
		self.generator_optimizer = optimizer["generator"]
		self.discriminator_optimizer = optimizer["discriminator"]

	def update_core(self):
		batch = self.get_iterator("main").next()
		batchsize = len(batch)

		# Train discriminator
		x_real = self.discriminator.wrap_array(batch)
		y_real = self.discriminator(x_real, self.stage, self.alpha)
		gradient = grad([y_real], [x_real], enable_double_backprop=True)[0]
		gradient_norm = sum(batch_l2_norm_squared(gradient)) / batchsize
		loss_grad = self.gamma * gradient_norm / 2
		z = self.generator.generate_latent(batchsize)
		mix_z = self.generator.generate_latent(batchsize) if self.mixing > random() else None
		x_fake = self.generator(z, self.stage, self.alpha, mix_z)
		y_fake = self.discriminator(x_fake, self.stage, self.alpha)
		loss_dis = ((sum((y_real - 1) ** 2) + sum(y_fake ** 2)) / 2 if self.lsgan else (sum(softplus(-y_real)) + sum(softplus(y_fake)))) / batchsize
		loss_dis += loss_grad
		x_fake.unchain_backward()
		self.discriminator.cleargrads()
		loss_dis.backward()
		self.discriminator_optimizer.update()

		# Train generator
		z = self.generator.generate_latent(batchsize)
		mix_z = self.generator.generate_latent(batchsize) if self.mixing > random() else None
		x_fake = self.generator(z, self.stage, self.alpha, mix_z)
		y_fake = self.discriminator(x_fake, self.stage, self.alpha)
		loss_gen = (sum((y_fake - 1) ** 2) / 2 if self.lsgan else sum(softplus(-y_fake))) / batchsize
		self.generator.cleargrads()
		loss_gen.backward()
		self.mapper_optimizer.update()
		self.generator_optimizer.update()

		report({"alpha": self.alpha})
		report({"loss (gen)": loss_gen})
		report({"loss (dis)": loss_dis})
		report({"loss (grad)": loss_grad})
		self.alpha = min(1.0, self.alpha + self.delta)
