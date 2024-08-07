import tensorflow as tf
import numpy as np

class AttentionConv(tf.keras.layers.Layer):
	def __init__(self, attention_kernel_size, regularizer=None, dropout_rate=0.0, softmax=False, **kwargs):
		super(AttentionConv, self).__init__(**kwargs)

		self.n_patches = 0
		self.n_w_patches = 0
		if type(attention_kernel_size) == int:
			self.attention_kernel_size = [attention_kernel_size, attention_kernel_size]
		else:
			self.attention_kernel_size = attention_kernel_size

		self.regularizer = regularizer
		self.dropout_rate = dropout_rate
		self.softmax = softmax
		self.attention_kernel = None
		self.ones = None
		self.normalize = None

	def call(self, x):
		if len(x.shape) == 4:
			w = x.shape[1]
			h = x.shape[2]
		else:
			w = x.shape[2]
			h = x.shape[3]

		if self.softmax:
			attention_kernel = self.attention_kernel / tf.math.sqrt(self.n_patches)
			attention_kernel = tf.nn.softmax(attention_kernel)
		else:
			attention_kernel = self.attention_kernel

		if self.dropout_rate > 0.0:
			attention_kernel = self.drop_layer(attention_kernel)

		attention_mat = tf.math.multiply(self.ones, attention_kernel)
		attention_mat = tf.unstack(attention_mat, axis=2)
		attention_mat = tf.concat(attention_mat, axis=1)
		attention_mat = tf.split(attention_mat, self.n_w_patches, 1)
		attention_mat = tf.concat(attention_mat, axis=0)
		attention_mat = attention_mat[0:w, 0:h]
		attention_mat = tf.repeat(tf.expand_dims(attention_mat, axis=-1), axis=-1, repeats=x.shape[-1])

		# Attention
		x1 = tf.math.multiply(x, attention_mat)
		x1 = self.normalize(x1)

		return x1

	def build(self, input_shape):
		if len(input_shape) == 4:
			self.n_patches = int(np.ceil(input_shape[1] / self.attention_kernel_size[0])) * int(np.ceil(input_shape[2] / self.attention_kernel_size[1]))
			self.n_w_patches = int(np.ceil(input_shape[1] / self.attention_kernel_size[0]))
		elif len(input_shape) == 5:
			self.n_patches = int(np.ceil(input_shape[2] / self.attention_kernel_size[0])) * int(np.ceil(input_shape[3] / self.attention_kernel_size[1]))
			self.n_w_patches = int(np.ceil(input_shape[2] / self.attention_kernel_size[0]))

		self.attention_kernel = self.add_weight("attention_kernel", shape=[self.n_patches,], initializer=tf.keras.initializers.Ones(),
	                              regularizer=self.regularizer, trainable=True, aggregation=tf.VariableAggregation.MEAN)

		self.ones = tf.ones([self.attention_kernel_size[0], self.attention_kernel_size[1], self.n_patches], dtype=tf.dtypes.float32)

		if self.dropout_rate > 0.0:
			self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)

		self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.n_patches = tf.cast(self.n_patches, dtype=tf.dtypes.float32)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'attention_kernel_size': self.attention_kernel_size,
			'regularizer': self.regularizer,
			'dropout_rate': self.dropout_rate,
			'softmax': self.softmax
		})
		return config


