import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked
from typing import Optional

class TripletBatchAllLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with semi-hard negative mining.
	The loss encourages the positive distances (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance
	among which are at least greater than the positive distance plus the
	margin constant (called semi-hard negative) in the mini-batch.
	If no such negative exists, uses the largest negative distance instead.
	See: https://arxiv.org/abs/1503.03832.
	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.
	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self, margin: FloatTensorLike = 1.0, norm=False, soft=False, adaptative=False, n_parts=0, n_iters=100, name: Optional[str] = None, **kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin
		self.norm = norm
		self.soft = soft
		self.adaptative = adaptative
		self.n_iters = n_iters
		self.n_parts = n_parts
		if self.adaptative:
			self.margins = margin * tf.ones([n_parts, 1, 1, 1])
			self.counts = tf.zeros([n_parts])
			self.n_iters = n_iters

	def call(self, y_true, y_pred):
		labels, embeddings = y_true, y_pred

		convert_to_float32 = (
				embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
		)
		precise_embeddings = (
			tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
		)

		if self.norm:
			precise_embeddings = tf.math.l2_normalize(precise_embeddings, axis=1)

		_shape = tf.shape(precise_embeddings)
		n = _shape[0]
		m = _shape[1]
		d = _shape[2]

		labels = tf.transpose(labels, [1, 0])
		labels = tf.repeat(labels, n, axis=0)
		hp_mask = tf.reshape(tf.expand_dims(labels, axis=1) == tf.expand_dims(labels, axis=2), shape=[-1])
		hn_mask = tf.reshape(tf.expand_dims(labels, axis=1) != tf.expand_dims(labels, axis=2), shape=[-1])

		dist = self.batch_dist(precise_embeddings)

		dist = tf.reshape(dist, shape=[-1])

		full_hp_dist = tf.reshape(tf.boolean_mask(dist, hp_mask), [n, m, -1, 1])
		full_hn_dist = tf.reshape(tf.boolean_mask(dist, hn_mask), [n, m, 1, -1])

		if self.soft:
			full_loss_metric = tf.reshape(tf.math.log1p(tf.math.exp(tf.subtract(full_hp_dist, full_hn_dist))),
			                              [n, -1])
		else:
			if self.adaptative:
				full_loss_metric = tf.reshape(tf.math.maximum((self.margins * tf.ones_like(full_hp_dist)) + tf.subtract(full_hp_dist, full_hn_dist), 0.0), [n, -1])
			else:
				full_loss_metric = tf.reshape(tf.math.maximum(self.margin + tf.subtract(full_hp_dist, full_hn_dist), 0.0), [n, -1])

		full_loss_metric_sum = tf.math.reduce_sum(full_loss_metric, axis=1)
		valid_triplets = tf.cast(tf.greater(full_loss_metric, 0.0), dtype=tf.dtypes.float32)
		full_loss_num = tf.math.reduce_sum(valid_triplets, axis=1)

		full_loss_metric_mean = full_loss_metric_sum / full_loss_num

		full_loss_metric_mean = tf.where(tf.not_equal(full_loss_num, 0.0), full_loss_metric_mean, tf.zeros_like(full_loss_metric_mean))

		if self.adaptative:
			margin_updates = tf.where(tf.equal(full_loss_num, 0.0), tf.ones_like(full_loss_metric_mean), tf.zeros_like(full_loss_metric_mean))
			self.counts = self.counts + margin_updates
			where = tf.where(tf.greater(self.counts, self.n_iters), tf.ones_like(self.counts), tf.zeros_like(self.counts))
			self.margins = self.margins + (tf.reshape(where, [-1, 1, 1, 1]) * 0.1)
			self.counts = tf.where(tf.greater(self.counts, self.n_iters), tf.zeros_like(self.counts), tf.ones_like(self.counts))

		if convert_to_float32:
			return tf.cast(full_loss_metric_mean, precise_embeddings.dtype)
		else:
			return full_loss_metric_mean

	def get_config(self):
		config = {
			"margin": self.margin,
			"norm": self.norm,
			"soft": self.soft,
			"adaptative": self.adaptative,
			"n_parts": self.n_parts,
			"n_iters": self.n_iters,
		}
		base_config = super().get_config()
		return {**base_config, **config}

	def batch_dist(self, x):
		x2 = tf.math.reduce_sum(tf.math.square(x), axis=2)
		dist = tf.expand_dims(x2, axis=2) + tf.expand_dims(x2, axis=1) - 2.0 * tf.linalg.matmul(x, tf.transpose(x,
		                                                                                                        [0, 2,
		                                                                                                         1]))
		dist = tf.math.maximum(dist, 0.0)
		error_mask = tf.math.less_equal(dist, 0.0)
		dist = tf.math.sqrt(dist + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)
		dist = tf.math.multiply(dist, tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32), )
		return dist
