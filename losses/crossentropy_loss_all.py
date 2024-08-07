import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked
from typing import Optional

class CrossentropyAllLoss(tf.keras.losses.Loss):
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

	def __init__(self):
		super().__init__(name="CrossentropyAllLoss", reduction=tf.keras.losses.Reduction.NONE)

	def call(self, y_true, y_pred):
		"""Computes the triplet loss with semi-hard negative mining.

				Args:
				  y_true: 1-D integer `Tensor` with shape [batch_size] of
					multiclass integer labels.
				  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
					be l2 normalized.
				  margin: Float, margin term in the loss definition.

				Returns:
				  triplet_loss: float scalar with dtype of y_pred.
				"""
		labels, probs = y_true, y_pred
		labels = tf.expand_dims(labels, axis=1)
		labels = tf.repeat(labels, probs.get_shape().as_list()[1], axis=1)

		convert_to_float32 = (
				probs.dtype == tf.dtypes.float16 or probs.dtype == tf.dtypes.bfloat16
		)
		precise_embeddings = (
			tf.cast(probs, tf.dtypes.float32) if convert_to_float32 else probs
		)
		epsilon_ = tf.constant(1e-07, precise_embeddings.dtype)
		precise_embeddings = tf.clip_by_value(precise_embeddings, epsilon_, 1. - epsilon_)
		crossentropy_loss = tf.math.negative(tf.math.multiply(labels, tf.math.log(precise_embeddings)))
		crossentropy_loss = tf.math.reduce_sum(crossentropy_loss, axis=-1)
		crossentropy_loss = tf.math.reduce_mean(crossentropy_loss, axis=1)

		if convert_to_float32:
			return tf.cast(crossentropy_loss, precise_embeddings.dtype)
		else:
			return crossentropy_loss

	def get_config(self):
		config = {}
		base_config = super().get_config()
		return {**base_config, **config}
