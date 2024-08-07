import tensorflow as tf
import numpy as np

class MatMul(tf.keras.layers.Layer):
	def __init__(self, bin_num=31, hidden_dim=256, input_dim=128, regularizer=None, **kwargs):
		super(MatMul, self).__init__(**kwargs)

		self.bin_num = bin_num
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.kernel = None
		self.regularizer = regularizer

	def build(self, input_shape):
		self.kernel = self.add_weight(name="matmul_kernel",
		                              shape=(self.bin_num, self.input_dim, self.hidden_dim),
		                              trainable=True, regularizer=self.regularizer,
		                              initializer=tf.keras.initializers.GlorotUniform(), aggregation=tf.VariableAggregation.MEAN)

	def call(self, x):
		return tf.matmul(x, self.kernel)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'bin_num': self.bin_num,
			'hidden_dim': self.hidden_dim,
			'input_dim': self.input_dim,
		})
		return config