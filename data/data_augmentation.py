import tensorflow as tf
from keras_cv.utils import fill_utils

def cutmix(images, labels, alpha=1.0, seed=None):
	"""Apply cutmix."""

	labels = tf.cast(labels, dtype=tf.dtypes.float32)
	input_shape = tf.shape(images)
	batch_size, image_height, image_width = (
		input_shape[0],
		input_shape[2],
		input_shape[3],
	)

	permutation_order = tf.random.shuffle(tf.range(0, input_shape[0]*input_shape[1], input_shape[1]), seed=seed)
	permutation_order = tf.reshape(tf.tile(tf.expand_dims(permutation_order, axis=1), [1, input_shape[1]]), [-1])

	sample_alpha = tf.random.gamma((batch_size,), 1.0, beta=alpha, seed=seed)
	sample_beta = tf.random.gamma((batch_size,), 1.0, beta=alpha, seed=seed)
	lambda_sample = sample_alpha / (sample_alpha + sample_beta)

	ratio = tf.math.sqrt(1 - lambda_sample)

	cut_height = tf.cast(
		ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
	)

	cut_height = tf.reshape(tf.tile(tf.expand_dims(cut_height, axis=1), [1, input_shape[1]]), [-1])

	cut_width = tf.cast(
		ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
	)

	cut_width = tf.reshape(tf.tile(tf.expand_dims(cut_width, axis=1), [1, input_shape[1]]), [-1])

	random_center_height = tf.random.uniform(
		shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
	)
	random_center_width = tf.random.uniform(
		shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
	)

	bounding_box_area = cut_height * cut_width
	lambda_sample = 1.0 - bounding_box_area / (image_height * image_width)
	lambda_sample = tf.cast(lambda_sample, dtype=tf.float32)

	# Deal with multiple frames.
	images = tf.reshape(images, [-1, input_shape[2], input_shape[3], input_shape[4]])
	random_center_height = tf.reshape(tf.tile(tf.expand_dims(random_center_height, axis=1), [1, input_shape[1]]), [-1])
	random_center_width = tf.reshape(tf.tile(tf.expand_dims(random_center_width, axis=1), [1, input_shape[1]]), [-1])

	images = fill_utils.fill_rectangle(
		images,
		random_center_width,
		random_center_height,
		cut_width,
		cut_height,
		tf.gather(images, permutation_order),
	)

	shape_label = tf.shape(labels)
	labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), [1, input_shape[1], 1]), [-1, shape_label[1]])

	cutout_labels = tf.gather(labels, permutation_order)

	lambda_sample = tf.reshape(lambda_sample, [-1, 1])
	labels = lambda_sample * labels + (1.0 - lambda_sample) * cutout_labels

	labels = tf.reduce_max(tf.reshape(labels, [input_shape[0], input_shape[1], -1]), axis=1)

	images = tf.reshape(images, [input_shape[0], input_shape[1], input_shape[2], input_shape[3], -1])

	return images, labels

def mixup(images, labels, alpha=0.2, seed=None):
	"""Apply mixup."""

	labels = tf.cast(labels, dtype=tf.dtypes.float32)
	batch_size = tf.shape(images)[0]
	permutation_order = tf.random.shuffle(tf.range(0, batch_size), seed=seed)

	sample_alpha = tf.random.gamma((batch_size,), 1.0, beta=alpha, seed=seed)
	sample_beta = tf.random.gamma((batch_size,), 1.0, beta=alpha, seed=seed)
	lambda_sample = sample_alpha / (sample_alpha + sample_beta)
	lambda_sample = tf.reshape(lambda_sample, [-1, 1, 1, 1, 1])


	mixup_images = tf.gather(images, permutation_order)

	images = lambda_sample * images + (1.0 - lambda_sample) * mixup_images

	labels_for_mixup = tf.gather(labels, permutation_order)

	lambda_sample = tf.reshape(lambda_sample, [-1, 1])
	labels = lambda_sample * labels + (1.0 - lambda_sample) * labels_for_mixup

	return images, labels
