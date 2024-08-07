import tensorflow as tf
import os
import numpy as np
import gc
class KNN:
	def __init__(self, gpu):
		self.gpu = gpu

	def predict(self, gallery_features, gallery_labels, probe_features, metric, k, p=2):
		nparts = int(np.ceil(gallery_features.shape[0] / 100000.0))
		predictions = np.zeros((probe_features.shape[0], k), dtype=np.int32)
		with tf.device('/GPU:0'):
			gallery = tf.constant(gallery_features)
			for sample_ix in range(probe_features.shape[0]):
				probe_sample = probe_features[sample_ix, :]
				distances = np.zeros((gallery.shape[0]), dtype=np.float32)
				for part_ix in range(nparts): # The gallery is split into parts to avoid OOM problems.
					i = part_ix * int(np.ceil(gallery.shape[0] / nparts))
					e = min((part_ix + 1) * int(np.ceil(gallery.shape[0] / nparts)), gallery.shape[0])
					if metric == 'L1': # sum(|x - y|)
						distances[i:e] = tf.reduce_sum(tf.abs(tf.subtract(gallery[i:e, :], probe_sample)), axis=1)
					elif metric == 'L2': # sqrt(sum((x - y)^2))
						distances[i:e] = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(gallery[i:e, :], probe_sample)), axis=1))
					elif metric == 'chebyshev': # max(|x - y|)
						distances[i:e] = tf.reduce_max(tf.abs(tf.subtract(gallery[i:e, :], probe_sample)), axis=1)
					elif metric == 'minkowski': # sum(|x - y|^p)^(1/p)
						distances[i:e] = tf.pow(tf.reduce_sum(tf.pow(tf.abs(tf.subtract(gallery[i:e, :], probe_sample)), p), axis=1), 1/p)
					else:
						distances[i:e] = tf.reduce_sum(tf.abs(tf.subtract(gallery[i:e, :], probe_sample)), axis=1)
				
				# nearest k points
				_, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
				top_k_label = tf.gather(gallery_labels, top_k_indices)
				predictions[sample_ix, :] = top_k_label #tf.unique_with_counts(top_k_label).y[tf.argmax(tf.unique_with_counts(top_k_label).count, output_type=tf.int32)]
			gc.collect()
		predictions = tf.squeeze(predictions)
		return predictions

	# def predict(self, gallery_features, gallery_labels, probe_features, metric, k, p=2):
	# 	distances = np.zeros((probe_features.shape[0], gallery_features.shape[0]), dtype=np.float32)
	# 	for part_ix in range(2): # The gallery is split into two parts to avoid OOM problems.
	# 		i = part_ix * int(np.ceil(gallery_features.shape[0] / 2))
	# 		e = min((part_ix + 1) * int(np.ceil(gallery_features.shape[0] / 2)), gallery_features.shape[0])
	# 		with tf.device('/GPU:0'):
	# 			gallery = tf.constant(gallery_features[i:e, :], dtype=tf.float32)
	# 			for sample_ix in range(probe_features.shape[0]):
	# 				sample = probe_features[sample_ix, :]
	# 				if metric == 'L1': # sum(|x - y|)
	# 					distances[sample_ix, i:e] = tf.reduce_sum(tf.abs(tf.subtract(gallery, sample)), axis=1)
	# 				elif metric == 'L2': # sqrt(sum((x - y)^2))
	# 					distances[sample_ix, i:e] = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(gallery, sample)), axis=1))
	# 				elif metric == 'chebyshev': # max(|x - y|)
	# 					distances[sample_ix, i:e] = tf.reduce_max(tf.abs(tf.subtract(gallery, sample)), axis=1)
	# 				elif metric == 'minkowski': # sum(|x - y|^p)^(1/p)
	# 					distances[sample_ix, i:e] = tf.pow(tf.reduce_sum(tf.pow(tf.abs(tf.subtract(gallery, sample)), p), axis=1), 1/p)
	# 				else:
	# 					distances[sample_ix, i:e] = tf.reduce_sum(tf.abs(tf.subtract(gallery, sample)), axis=1)
	#
	# 	# nearest k points
	# 	predictions = np.zeros((probe_features.shape[0]), dtype=np.int32)
	# 	with tf.device('/GPU:0'):
	# 		i = 0
	# 		e = 0
	# 		while e < distances.shape[0]: # The gallery is split into two parts to avoid OOM problems.
	# 			e = min(i + 2048, distances.shape[0])
	# 			_, top_k_indices = tf.nn.top_k(tf.negative(distances[i:e, :]), k=k)
	# 			top_k_label = tf.gather(gallery_labels, top_k_indices)
	# 			predictions[i:e] = tf.map_fn(lambda x: tf.unique_with_counts(x).y[tf.argmax(tf.unique_with_counts(x).count, output_type=tf.int32)], top_k_label)
	# 			i = e
	#
	# 	return predictions
