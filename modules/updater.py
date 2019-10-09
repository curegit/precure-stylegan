from chainer import grad
from chainer.reporter import report
from chainer.training import StandardUpdater
from chainer.functions import sum, batch_l2_norm_squared, softplus

# Updater for Style GAN
class StyleGanUpdater(StandardUpdater):

	def __init__(self, generator, discriminator, iterator, optimizer, device, stage, alpha=1.0, delta=0.002, gamma=10):
		super().__init__(iterator, optimizer, device=device)
		self.alpha = alpha
		self.delta = delta
		self.gamma = gamma
		self.stage = stage
		self.generator = generator
		self.discriminator = discriminator
		self.mapper_optimizer = optimizer["mapper"]
		self.generator_optimizer = optimizer["generator"]
		self.discriminator_optimizer = optimizer["discriminator"]

	def update_core(self):
		x_real = self.get_iterator("main").next()
		batch = len(x_real)

		# Train discriminator
		y_real = self.discriminator(x_real, self.stage, self.alpha)
		gradient = grad([y_real], [x_real], enable_double_backprop=True)[0]
		gradient_norm = sum(batch_l2_norm_squared(gradient)) / batch
		loss_gamma = self.gamma * gradient_norm / 2
		z = self.generator.generate_latent(batch)
		x_fake = self.generator(z, self.stage, self.alpha)
		y_fake = self.discriminator(x_fake, self.stage, self.alpha)
		loss_dis = sum(softplus(-y_real)) / batch
		loss_dis += sum(softplus(y_fake)) / batch
		loss_dis += loss_gamma
		x_fake.unchain_backward()
		self.discriminator.cleargrads()
		loss_dis.backward()
		self.discriminator_optimizer.update()

		# Train generator
		z = self.generator.generate_latent(batch)
		x_fake = self.generator(z, self.stage, self.alpha)
		y_fake = self.discriminator(x_fake, self.stage, self.alpha)
		loss_gen = sum(softplus(-y_fake)) / batch
		self.generator.cleargrads()
		loss_gen.backward()
		self.mapper_optimizer.update()
		self.generator_optimizer.update()

		report({"alpha": self.alpha})
		report({"loss_gen": loss_gen})
		report({"loss_dis": loss_dis})
		self.alpha += self.delta
