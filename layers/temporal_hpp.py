import tensorflow as tf

class TemporalHPP(tf.keras.layers.Layer):
	def __init__(self, n_filters, bins=None, regularizer=None, activation_func=tf.nn.leaky_relu, reduction='both', **kwargs):
		super(TemporalHPP, self).__init__(**kwargs)
		self.n_filters = n_filters
		if bins is None:
			self.bins = [1, 2, 4, 8, 16]
		else:
			self.bins = bins

		self.regularizer = regularizer
		self.convs = []
		self.activation_func = activation_func
		self.reduction = reduction

	def call(self, x):
		# HPP
		n, f, h, w, c = x.shape
		features = list()
		for i in range(len(self.convs)):
			hpp = self.convs[i](x)
			hpp = tf.math.reduce_mean(hpp, axis=1) + tf.math.reduce_max(hpp, axis=1)
			n_patches = h // self.bins[i]
			hpp = tf.reshape(hpp, [-1, n_patches, self.n_filters])
			features.append(hpp)

		features = tf.concat(features, axis=1)
		return features

	def build(self, input_shape):
		for bin in self.bins:
			self.convs.append(tf.keras.layers.Conv3D(self.n_filters, kernel_size=(bin, bin, input_shape[3]), activation=self.activation_func, padding='valid', use_bias=False,
											groups=input_shape[-1], strides=(bin, bin, input_shape[3]), data_format='channels_last', kernel_regularizer=self.regularizer))

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


