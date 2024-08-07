import tensorflow as tf
import numpy as np

class CosineLrScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, warmup_steps=0, hold=0, total_steps=0, start_lr=0.0, target_lr=1e-3):
		super().__init__()

		self.warmup_steps = warmup_steps
		self.hold = hold
		self.total_steps = total_steps
		self.start_lr = start_lr
		self.target_lr = target_lr
		self.pi = tf.constant(np.pi, dtype=tf.dtypes.float32)
		self.current_lr = start_lr

	def __call__(self, step):
		step = tf.cast(step, dtype=tf.dtypes.float32)
		learning_rate = 0.5 * self.target_lr * (
				1 + tf.math.cos(self.pi * (step - self.warmup_steps - self.hold) / tf.cast(self.total_steps - self.warmup_steps - self.hold, dtype=tf.dtypes.float32)))

		# Target LR * progress of warmup (=1 at the final warmup step)
		warmup_lr = self.target_lr * (step / self.warmup_steps)

		# Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
		# i.e. warm up if we're still warming up and use cosine decayed lr otherwise
		if self.hold > 0:
			learning_rate = tf.where(step > self.warmup_steps + self.hold,
									 learning_rate, self.target_lr)

		learning_rate = tf.where(step < self.warmup_steps, warmup_lr, learning_rate)
		self.current_lr = learning_rate
		return learning_rate