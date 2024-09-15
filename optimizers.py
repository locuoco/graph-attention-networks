from keras import optimizers
from keras import ops

class Adan(optimizers.Optimizer):
	# Adaptive Nesterov Momentum Algorithm
	# see X. Xie et al. (2022)
	def __init__(
		self,
		learning_rate=0.001,
		beta_1=0.02,
		beta_2=0.08,
		beta_3=0.01,
		epsilon=1e-7,
		weight_decay=None,
		clipnorm=None,
		clipvalue=None,
		global_clipnorm=None,
		use_ema=False,
		ema_momentum=0.99,
		ema_overwrite_frequency=None,
		loss_scale_factor=None,
		gradient_accumulation_steps=None,
		name='adan',
		**kwargs,
	):
		super().__init__(
			learning_rate=learning_rate,
			name=name,
			clipnorm=clipnorm,
			clipvalue=clipvalue,
			global_clipnorm=global_clipnorm,
			use_ema=use_ema,
			ema_momentum=ema_momentum,
			ema_overwrite_frequency=ema_overwrite_frequency,
			loss_scale_factor=loss_scale_factor,
			gradient_accumulation_steps=gradient_accumulation_steps,
			**kwargs,
		)
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.beta_3 = beta_3
		self.epsilon = epsilon
		self.weight_decay = weight_decay

	def build(self, var_list):
		# For each model variable, create the optimizer variable associated with it.
		if self.built:
			return
		super().build(var_list)
		self.ms_ = []
		self.vs_ = []
		self.ns_ = []
		self.prevgs_ = []
		for var in var_list:
			self.ms_.append(
				self.add_variable_from_reference(reference_variable=var, name='m')
			)
			self.vs_.append(
				self.add_variable_from_reference(reference_variable=var, name='v')
			)
			self.ns_.append(
				self.add_variable_from_reference(reference_variable=var, name='n')
			)
			self.prevgs_.append(
				self.add_variable_from_reference(reference_variable=var, name='prevg')
			)

	def update_step(self, grad, var, learning_rate):
		lr = ops.cast(learning_rate, var.dtype)
		grad = ops.cast(grad, var.dtype)
		local_step = self.iterations

		m = self.ms_[self._get_variable_index(var)]
		v = self.vs_[self._get_variable_index(var)]
		n = self.ns_[self._get_variable_index(var)]
		prevg = self.prevgs_[self._get_variable_index(var)]

		diff = ops.where(local_step == 0, 0, grad - prevg)

		self.assign(m, ops.where(
			local_step == 0,
			grad,
			ops.add(m, ops.multiply(ops.subtract(grad, m), self.beta_1)),
		))

		self.assign(v, ops.where(
			local_step <= 1,
			diff,
			ops.add(v, ops.multiply(ops.subtract(diff, v), self.beta_2)),
		))

		self.assign(n, ops.where(
			local_step == 0,
			ops.square(grad),
			ops.add(n, ops.multiply(ops.subtract(ops.square(ops.add(grad, ops.multiply(diff, 1 - self.beta_2))), n), self.beta_3)),
		))

		eta = lr / (ops.sqrt(n) + self.epsilon)

		if self.weight_decay is not None:
			self.assign(
				var, ops.divide(ops.subtract(var, ops.multiply(eta, ops.add(m, ops.multiply(v, 1 - self.beta_2)))), 1 + self.weight_decay * lr)
			)
		else:
			self.assign_sub(
				var, ops.multiply(eta, ops.add(m, ops.multiply(v, 1 - self.beta_2)))
			)

		self.assign(prevg, grad)

	def get_config(self):
		config = super().get_config()
		config.update(
			{
				'beta_1': self.beta_1,
				'beta_2': self.beta_2,
				'beta_3': self.beta_3,
				'epsilon': self.epsilon,
				'weight_decay': self.weight_decay,
			}
		)
		return config









