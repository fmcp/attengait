import tensorflow as tf


class AttentionHPP(tf.keras.layers.Layer):
	def __init__(self, regularizer=None, dropout_rate=0.0, softmax=False, **kwargs):
		super(AttentionHPP, self).__init__(**kwargs)

		self.n_patches = 0
		self.regularizer = regularizer
		self.dropout_rate = dropout_rate
		self.softmax = softmax
		self.attention_kernel = None
		self.normalize = None

	def call(self, x):
		if self.softmax:
			attention_kernel = self.attention_kernel / tf.math.sqrt(self.n_patches)
			attention_kernel = tf.nn.softmax(attention_kernel)
		else:
			attention_kernel = self.attention_kernel

		if self.dropout_rate > 0.0:
			attention_kernel = self.drop_layer(attention_kernel)

		# Attention
		attention_mat = tf.repeat(tf.expand_dims(attention_kernel, axis=-1), axis=-1, repeats=x.shape[-1])
		x1 = tf.math.multiply(x, attention_mat)
		x1 = self.normalize(x1)

		return x1

	def build(self, input_shape):
		self.n_patches = input_shape[1]
		self.attention_kernel = self.add_weight("attention_sum_kernel",
		                                        shape=[self.n_patches, ], initializer=tf.keras.initializers.Ones(),
		                                        regularizer=self.regularizer, trainable=True,
		                                        aggregation=tf.VariableAggregation.MEAN)

		if self.dropout_rate > 0.0:
			self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)

		self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.n_patches = tf.cast(self.n_patches, dtype=tf.dtypes.float32)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'regularizer': self.regularizer,
			'dropout_rate': self.dropout_rate,
		    'softmax': self.softmax,
		})
		return config
