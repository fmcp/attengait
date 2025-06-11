import os
import numpy as np
import random
import tensorflow as tf
from operator import itemgetter
import copy
import cv2
from cvbase.optflow.visualize import flow2rgb
import h5py
from data.data_augmentation import cutmix
from data.data_augmentation import mixup

random.seed(10)

class DataGeneratorGait:
	"""
	A class used to generate data for training/testing CNN gait nets

	Attributes
	----------
	dim : tuple
		dimensions of the input data
	n_classes : int
		number of classes
	...
	"""

	def __init__(self, dataset_info, batch_size=128, mode='train', labmap=[],
				 camera=None, datadir="", augmentation=True,
				 keep_data=True, max_frames=30, p=8, k=16, cut=False, diffFrames=False,
				 pk=False, crossentropy_loss=False, combine_outputs=False,
				 aux_losses=False, random_frames=False, substract_mean=False, multi_gpu=False, repeat=False,
				 split_crossentropy=True):
		'Initialization'

		self.camera = camera
		self.datadir = datadir
		self.empty_files = []
		self.batches = []
		self.batches_test = []
		self.mode = mode
		self.keep_data = keep_data
		self.data = {}
		self.epoch = 0
		self.p = int(p)
		self.k = int(k)
		self.different_frames = diffFrames
		self.crossentropy_loss = crossentropy_loss
		self.test_iterator = None
		self.pk = pk
		self.combine_outputs = combine_outputs
		self.random_init_frame = True
		self.n_frames = max_frames
		self.random_frames = random_frames
		self.mean = None
		self.substract_mean = substract_mean
		self.aux_losses = aux_losses
		self.multi_gpu = multi_gpu
		self.repeat = repeat
		self.split_crossentropy = split_crossentropy

		if mode == 'train':
			self.set = 1
			self.augmentation = augmentation
		elif mode == 'val':
			self.set = 2
			self.augmentation = False
		elif mode == 'trainval' or mode == 'trainvaltest':
			self.set = -1
			self.augmentation = augmentation
			self.shuffle_data = False
		else:
			self.set = 3
			self.augmentation = False

		if mode == 'train':
			self.gait = dataset_info['gait']
			self.file_list = dataset_info['file']
			self.labels = dataset_info['label']
			self.videoId = dataset_info['videoId']
		elif mode == 'val' or mode == 'valtest':
			self.gait = dataset_info['gait']
			self.file_list = dataset_info['file']
			self.labels = dataset_info['label']
			self.videoId = dataset_info['videoId']
		elif mode == 'test':
			self.gait = dataset_info['gait']
			self.file_list = dataset_info['file']
			self.labels = dataset_info['label']
			self.videoId = dataset_info['videoId']
		else:
			self.gait = dataset_info['gait']
			self.file_list = dataset_info['file']
			self.labels = dataset_info['label']
			self.videoId = dataset_info['videoId']

		if len(np.unique(dataset_info['label'])) == 74:
			# Remove subject 5
			pos = np.where(self.labels != 5)[0]
			self.gait = self.gait[pos]
			self.file_list = list(itemgetter(*list(pos))(self.file_list))
			self.labels = self.labels[pos]
			self.videoId = self.videoId[pos]

		if self.mode == 'valtest':
			# We have to change the mode to run a normal test for the acc callback.
			self.mode = 'test'

		nclasses = len(np.unique(dataset_info['label']))
		self.nclasses = nclasses
		self.ulabs = np.unique(self.labels)

		if nclasses == 20000 or nclasses == 6000:
			self.camera = None
			self.cameras = [None] * len(self.labels)
		else:
			if "cam" in dataset_info.keys():
				self.cameras = dataset_info['cam']
			else:
				self.cameras = self.__derive_camera()

		self.max_frames = max_frames

		self.__remove_empty_files()
		if self.camera is not None:
			self.__keep_camera_samples()

		self.batch_size = np.min((batch_size, len(self.file_list))) # Deal with less number of samples than batch size

		self.compressFactor = dataset_info['compressFactor']
		self.cut = cut
		self.dim = (3, 64, 64)
		self.ugaits = np.unique(self.gait)
		self.labmap = labmap

		# Create mapping for labels
		if labmap is not None:
			self.labmap_tf = np.zeros(len(list(self.labmap.keys())), dtype=np.int32)
			for key in list(self.labmap.keys()):
				self.labmap_tf[self.labmap[key]] = key
		else:
			self.labmap_tf = None

		# Init some variables
		self.videos_pxk = []
		ulabs_ = np.unique(self.labels)
		for label in ulabs_:
			ids_ = np.where(self.labels == label)[0]
			self.videos_pxk.append(ids_)

		self.aug_rand = 0.25
		self.dataset = self.create_dataset()

	def __len__(self):
		'Number of batches per epoch'
		size = len(self.file_list)
		if self.pk:
			n_step = int(np.ceil((len(self.ulabs) / self.p) / (self.batch_size / (self.p * self.k)))) # The last part is to deal with multigpu
		else:
			n_step = int(size / self.batch_size)
		return n_step

	def __getitemvideoid__(self, index):
		"""Generate one batch of data"""
		return next(self.test_iterator)

	def __derive_camera(self):
		cameras = []
		for file in self.file_list:
			parts = file.split("-")
			# 001-nm-01-000.h5
			parts2 = parts[3].split(".")
			cameras.append(int(parts2[0]))

		return cameras

	def __remove_empty_files(self):
		gait_ = []
		file_list_ = []
		labels_ = []
		videoId_ = []
		cameras_ = []
		max_frames = self.max_frames

		for i in range(len(self.file_list)):
			f = h5py.File(os.path.join(self.datadir, self.file_list[i]), 'r')
			if f["data"].shape[0] >= max_frames:
				gait_.append(self.gait[i])
				file_list_.append(self.file_list[i])
				labels_.append(self.labels[i])
				videoId_.append(self.videoId[i])
				cameras_.append(self.cameras[i])
			f.close()

		self.gait = gait_
		self.file_list = file_list_
		self.labels = labels_
		self.videoId = videoId_
		self.cameras = cameras_

	def __keep_camera_samples(self):
		gait_ = []
		file_list_ = []
		labels_ = []
		videoId_ = []
		cameras_ = []
		for i in range(len(self.file_list)):
			for j in range(len(self.camera)):
				cam_str = "{:03d}".format(self.camera[j])
				if cam_str in self.file_list[i] and self.file_list[i] not in file_list_:
					gait_.append(self.gait[i])
					file_list_.append(self.file_list[i])
					labels_.append(self.labels[i])
					videoId_.append(self.videoId[i])
					cameras_.append(self.cameras[i])

		self.gait = gait_
		self.file_list = file_list_
		self.labels = labels_
		self.videoId = videoId_
		self.cameras = cameras_

	def create_dataset(self):
		AUTOTUNE = tf.data.AUTOTUNE
		if self.mode == 'test' or self.mode == 'trainval':
			dataset = tf.data.Dataset.from_tensor_slices((self.file_list, self.labels, self.videoId, self.cameras))
			options = tf.data.Options()
			options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
			dataset = dataset.with_options(options)
		else:
			dataset = tf.data.Dataset.from_tensor_slices((np.arange(len(np.unique(self.labels)), dtype=np.int32), np.unique(self.labels).astype(np.int32)))
			dataset = dataset.shuffle(buffer_size=len(np.unique(self.labels)), reshuffle_each_iteration=True)
			dataset = dataset.interleave(lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.k), cycle_length=self.p, block_length=self.k, deterministic=True, num_parallel_calls=AUTOTUNE)

		dataset = dataset.map(self.load_data(), num_parallel_calls=AUTOTUNE)
		dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=self.multi_gpu)
		dataset = dataset.map(self.augment_data(), num_parallel_calls=AUTOTUNE)
		dataset = dataset.unbatch().batch(batch_size=self.batch_size, drop_remainder=self.multi_gpu > 1)
		dataset = dataset.map(self.prepare_data(self.crossentropy_loss, self.combine_outputs), num_parallel_calls=AUTOTUNE)

		dataset = dataset.prefetch(buffer_size=AUTOTUNE)

		if self.mode == "train":
			dataset = dataset.repeat()
		else:
			self.test_iterator = dataset.as_numpy_iterator()

		return dataset

	def restart_test_iterator(self):
		self.test_iterator = self.dataset.as_numpy_iterator()

	def prepare_data(self, crossentropy_loss_, combine_outputs_):
		@tf.function
		def prepare_data_fun(sample, label, labels_one_hot, videoId=None, camera=None):
			if self.mode == 'train':
				label = tf.expand_dims(label, axis=-1)

			lab = label

			if self.different_frames:
				n_frames_ = np.random.randint(16, self.max_frames)
				sample = sample[:, 0:n_frames_, :, :, :]

			if self.split_crossentropy:
				lab_cross = labels_one_hot
			else:
				lab_cross = tf.squeeze(label)

			if crossentropy_loss_:
				label = {"encode": label, "probs": lab_cross}

			sample = {'input_1': sample}

			if self.aux_losses:
				label["aux_a"] = lab
				label["aux_b"] = lab

			if combine_outputs_:
				if type(label) is dict:
					label["combined_output"] = lab
					label["probs_combined_output"] = lab_cross
				else:
					label = {"encode": label, "combined_output": label, "probs_combined_output": lab_cross}

			if videoId is not None:
				return sample, label, videoId, camera
			else:
				return sample, label

		return prepare_data_fun

	def load_data(self):
		@tf.function
		def load_data_fun(index_data, index_label, videoId=None, camera=None):
			sample, label = tf.py_function(self.load_image, [index_data, index_label], [tf.dtypes.float32, tf.dtypes.int32])
			sample = tf.cast(sample, dtype=tf.float32)

			if self.substract_mean:
				sample = sample - self.mean

			sample = sample / 255.0
			sample = tf.clip_by_value((sample - 0.5) * 2, -1., 1.)

			if self.labmap is not None:
				label = tf.squeeze(tf.where(tf.equal(self.labmap_tf, label)))

			if videoId is not None:
				if camera is None:
					camera = index_data
				return sample, label, videoId, camera
			else:
				return sample, label

		return load_data_fun

	def __load_file(self, file_path):
		f = h5py.File(os.path.join(self.datadir, file_path), 'r')
		data = np.zeros(f['data'].shape, dtype=f['data'].dtype)
		f['data'].read_direct(data)
		f.close()
		return data

	def load_image(self, index_sample, label):
		if self.mode == 'test' or self.mode == 'trainval':
			filepath = index_sample.numpy().decode('UTF-8')
		else:
			filepath = self.file_list[np.random.choice(self.videos_pxk[index_sample], 1)[0]]

		if filepath in self.data:
			sample = self.data[filepath]
		else:
			sample = self.__load_file(filepath)
			if self.cut:
				sample_final = np.zeros((sample.shape[0], sample.shape[1], sample.shape[2], 3), dtype=np.uint8)
				for i in range(sample.shape[0]):
					sample_final[i, :, :, :] = cv2.resize(sample_final[i, :, 10:54, :], (64, 64))
				sample = sample_final

			if self.keep_data:
				self.data[filepath] = copy.deepcopy(sample)

		if self.repeat:
			n_repeats = int(max(np.ceil(30.0 / sample.shape[0]), 1.0))
			sample = np.repeat(sample, n_repeats, axis=0)

		return self.extract_frames(sample), label

	def extract_frames(self, sample):
		if self.mode == 'train':
			# Shuffle frames.
			if self.random_frames:
				sample = tf.random.shuffle(sample)
				init_pos = 0
			else:
				if self.random_init_frame:
					init_pos = np.random.randint(low=0, high=max(sample.shape[0] - self.n_frames, 0) + 1, size=1)[0]
				else:
					init_pos = np.max((sample.shape[0] // 2) - (self.n_frames // 2), 0)

			# Keep n_frames.
			pos = np.arange(start=init_pos, stop=min(init_pos + self.n_frames, sample.shape[0]), step=1)
			sample = tf.gather(sample, pos, axis=0)

		sample = tf.cast(sample, dtype=tf.float32)

		return sample

	def augment_data(self):
		@tf.function
		def augment_data_fun(samples, labels, videoId=None, camera=None):
			if self.nclasses == 50:
				labels_one_hot = tf.one_hot(labels - 74, self.nclasses)
			elif self.nclasses == 5154:
				labels_one_hot = tf.one_hot(labels - 5153, self.nclasses)
			else:
				labels_one_hot = tf.one_hot(labels, self.nclasses)

			if videoId is not None:
				return samples, labels, labels_one_hot, videoId, camera
			else:
				rand_number = tf.random.uniform([3], minval=0, maxval=1)

				if rand_number[0] < self.aug_rand:
					samples, labels_one_hot = mixup(samples, labels_one_hot)
				if rand_number[1] < self.aug_rand:
					samples, labels_one_hot = cutmix(samples, labels_one_hot)

				return samples, labels, labels_one_hot

		return augment_data_fun
