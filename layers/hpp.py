import tensorflow as tf

class HPP(tf.keras.layers.Layer):
	def __init__(self, n_filters, bins=None, regularizer=None, activation_func=tf.nn.leaky_relu, reduction='both', **kwargs):
		super(HPP, self).__init__(**kwargs)
		self.n_filters = n_filters
		if bins is None:
			self.bins = [1, 2, 4, 8, 16]
		else:
			self.bins = bins

		self.regularizer = regularizer
		self.convs = []
		self.activation_func = activation_func
		self.reduction = reduction
		self.reduction_func = None

	def call(self, x):
		# HPP
		x = self.reduction_func(x)
		features = list()
		for i in range(len(self.convs)):
			hpp = self.convs[i](x)
			n_patches = x.shape[1] // self.bins[i]
			hpp = tf.reshape(hpp, [-1, n_patches, self.n_filters])
			features.append(hpp)

		features = tf.concat(features, axis=1)
		return features

	def build(self, input_shape):
		for bin in self.bins:
			self.convs.append(tf.keras.layers.Conv2D(self.n_filters, kernel_size=(bin, input_shape[3]), activation=self.activation_func, padding='valid', use_bias=False,
											groups=input_shape[-1], strides=(bin, bin), data_format='channels_last', kernel_regularizer=self.regularizer))

		if self.reduction == 'both':
			self.reduction_func = lambda x: tf.math.reduce_mean(x, axis=1) + tf.math.reduce_max(x, axis=1)
		else:
			self.reduction_func = lambda x: tf.math.reduce_mean(x, axis=1)

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'n_filters': self.n_filters,
			'bins': self.bins,
			'regularizer': self.regularizer,
			'activation_func': self.activation_func,
			'reduction': self.reduction,
		})
		return config


