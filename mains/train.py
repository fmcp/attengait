import sys
import os
import time
import numpy as np
import os.path as osp
import pathlib

np.object = object    # Fix for deepdish

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")

# --------------------------------
import tensorflow as tf

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # gpu_rate # TODO
config.gpu_options.polling_inactive_delay_msecs = 50

graph = tf.Graph()
graph.as_default()
#
session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
tf.compat.v1.keras.backend.set_session(session)
# --------------------------------

from tensorflow.keras import optimizers

import deepdish as dd
from data.data_generator import DataGeneratorGait
from utils.utils import find_latest_file_model
import tensorflow_addons as tfa
from callbacks.cosine_lr_scheduler import CosineLrScheduler

def lr_scheduler(epoch, lr):
	if epoch % 2000 == 0 and epoch > 0:
		new_lr = lr * 0.5
	else:
		new_lr = lr

	tf.summary.scalar('learning rate', data=new_lr, step=epoch)

	return new_lr

def lr_scheduler_oumvlp(epoch, lr):
	if epoch % 500 == 0 and epoch > 0:
		new_lr = lr * 0.5
	else:
		new_lr = lr

	tf.summary.scalar('learning rate', data=new_lr, step=epoch)

	return new_lr


# ===============================================================
def trainGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", experfix="of", nclasses=0, lr=0.0001,
				 experdirbase=".", epochs=15, batchsize=150, optimizer="Adam",
				 extra_epochs=0, cameras=None,
				 sufix=None, margin=0.2, pk=0, p=8, k=16, weight_decay=0.00001, cut=False, diffFrames=False,
				 save_epoch=100, augmentation=False,
				 dropout=0, attention_drop_rate=0.25, crossentropy_weight=0.0,
				 norm_triplet_loss=False, soft_triplet_loss=False, kernel_regularizer=False,
                 lr_sched=False, verbose=0, split_crossentropy=False,
                 shared_weights_crossentropy=False, combine_outputs=False, model_size="small",
                 init_net="", softmax_attention=False, random_frames=False,
                 multi_gpu=0, adaptative_triplet_loss=False, substract_mean=False, reduction='both', combined_output_length=16,
                 repeat_data=False):

	"""
	Trains a CNN for gait recognition
	:param datadir: root dir containing dd files
	:param experfix: string to customize experiment name
	:param nclasses: number of classes
	:param lr: starting learning rate
	:param epochs: max number of epochs
	:param batchsize: integer
	:param optimizer: options like {SGD, Adam,...}
	:param logdir: path to save Tensorboard info
	:param verbose: integer
	:return: model, experdir, accuracy
	"""

	input_shape = (None, 64, 64, 3)

#	weight_decay = weight_decay #0.0005
	momentum = 0.9

	if pk:
		batchsize = p * k

	if multi_gpu > 0:
		batchsize = batchsize * multi_gpu
		lr = lr * multi_gpu
		weight_decay = weight_decay

	infix = "op" + optimizer

	infix = infix + "_wd" + str(weight_decay)

	if norm_triplet_loss:
		infix = infix + "_norm_tl"
		
	if soft_triplet_loss:
		infix = infix + "_soft_tl"

	if lr_sched:
		infix = infix + "_lr_sched"

	if split_crossentropy:
		infix = infix + "_split_crossentropy"

	if shared_weights_crossentropy:
		infix = infix + "_shared_weights_crossentropy"

	if combine_outputs:
		infix = infix + "_combine_outputs" + str(combined_output_length)

	if adaptative_triplet_loss:
		infix = infix + "_adap_tl"

	if substract_mean:
		infix = infix + "_sub_mean"

	if reduction != 'both':
		infix = infix + "red_max"

	if model_size == "small":
		n_filters = [32, 64, 128, 256]
	elif model_size == "medium":
		n_filters = [64, 128, 256, 256]
	elif model_size == "large":
		n_filters = [128, 256, 512, 256]
	else:
		n_filters = [32, 64, 128, 256]

	infix = infix + "_ms-" + model_size

	if softmax_attention:
		infix = infix + "_softmax_attention"

	if repeat_data:
		infix = infix + "_rep_data"

	# Create a TensorBoard instance with the path to the logs directory
	if nclasses > 999:
		subdir = experfix + '_attengait' + '_N{:05d}_{}_bs{:03d}_lr{:0.6f}'.format(nclasses,
																										   infix,
																										   batchsize,
																										   lr)  # To be customized
	else:
		subdir = experfix + '_attengait' + '_N{:03d}_{}_bs{:03d}_lr{:0.6f}'.format(nclasses, infix, batchsize, lr)  # To be customized
	if sufix is not None:
		subdir = subdir + "_" + sufix
	if diffFrames:
		subdir = subdir + "_" + "diffFrames"
	if cut:
		subdir = subdir + "_" + "cut"

	if crossentropy_weight > 0.0:
		subdir = subdir + "_crossentropy" + str(crossentropy_weight)

	experdir = osp.join(experdirbase, subdir)
	if verbose > 0:
		print(experdir)
	if not osp.exists(experdir):
		os.makedirs(experdir)

	# Tensorboard
	experdir_tensorboard = osp.join(experdir, "tensorboard")
	if not osp.exists(experdir_tensorboard):
		os.makedirs(experdir_tensorboard)

	train_summary_writer = tf.summary.create_file_writer(experdir_tensorboard)

	from nets.attengait import GaitSetTransformer as OurModel
	model = OurModel(experdir, reduce_channel=False, batch_size=batchsize)

	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	if nclasses == 74:
		data_folder = osp.join(datadir, 'tfimdb_casiab_N074_train_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_casiab_N074_train_of30_64x64.h5')
	elif nclasses == 50:
		data_folder = osp.join(datadir, 'tfimdb_casiab_N050_ft_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_casiab_N050_ft_of30_64x64.h5')
	elif nclasses == 5153:
		data_folder = osp.join(datadir, 'tfimdb_oumvlp_N5153_train_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_oumvlp_N5153_train_of30_64x64.h5')
	elif nclasses == 5154:
		data_folder = osp.join(datadir, 'tfimdb_oumvlp_N5154_ft_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_oumvlp_N5154_ft_of30_64x64.h5')
	elif nclasses == 20000:
		data_folder = osp.join(datadir, 'tfimdb_grew_N20000_train_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_grew_N20000_train_of30_64x64.h5')
	elif nclasses == 6000:
		data_folder = osp.join(datadir, 'tfimdb_grew_N06000_ft_of30_64x64')
		info_file = osp.join(datadir, 'tfimdb_grew_N06000_ft_of30_64x64.h5')
	else:
		sys.exit(0)

	dataset_info = dd.io.load(info_file)

	# Find label mapping for training
	if nclasses > 0:
		ulabels = np.unique(dataset_info['label'])
		# Create mapping for labels
		labmap = {}
		for ix, lab in enumerate(ulabels):
			labmap[int(lab)] = ix
	else:
		labmap = None

	# Data generators
	train_generator = DataGeneratorGait(dataset_info, batch_size=batchsize, mode='train', labmap=labmap,
	                                    datadir=data_folder, camera=cameras, p=p, k=k, cut=cut,
	                                    diffFrames=diffFrames, random_frames=random_frames,
	                                    augmentation=augmentation, crossentropy_loss=crossentropy_weight > 0.0,
	                                    pk=pk, combine_outputs=combine_outputs,
	                                    substract_mean=substract_mean,
	                                    aux_losses=False,
	                                    multi_gpu=multi_gpu > 0,
	                                    repeat=repeat_data,
	                                    split_crossentropy=split_crossentropy)


	if lr_sched:
		lr_ = CosineLrScheduler(20 * train_generator.__len__(), 0, (epochs+extra_epochs) * train_generator.__len__(), 0.1*lr, lr)
	else:
		lr_ = lr

	optimfun = optimizers.Adam(learning_rate=lr_)
	if optimizer != "Adam":
		if optimizer == "SGD":
			optimfun = optimizers.SGD(lr=lr, momentum=momentum)
		elif optimizer == "AMSGrad":
			optimfun = optimizers.Adam(lr=lr, amsgrad=True)
		elif optimizer == "AdamW":
			lr_ = tf.keras.optimizers.schedules.CosineDecay(lr, epochs * train_generator.__len__(), alpha=1.0/(lr/0.00001))
			weight_decay_ = tf.keras.optimizers.schedules.CosineDecay(weight_decay, epochs * train_generator.__len__(), alpha=0.1)
			optimfun = tfa.optimizers.AdamW(learning_rate=lr_, amsgrad=False, weight_decay=weight_decay_)
			weight_decay = 0
		elif optimizer == "AdaBelief":
			optimfun = tfa.optimizers.AdaBelief(learning_rate=lr, total_steps=epochs * train_generator.__len__(), warmup_proportion=0.1, min_lr=1e-5, rectify=True)
		elif optimizer == "LAMB":
			optimfun = tfa.optimizers.LAMB(learning_rate=lr_, weight_decay=weight_decay)
			weight_decay = 0
		else:
			optimfun = eval("optimizers." + optimizer + "(lr=initialLR)")

	if optimizer == "AdamW":
		weight_decay = 0.0

	# Prepare model
	pattern_file = "model-state-{:04d}.hdf5"
	previous_model = find_latest_file_model(experdir, pattern_file, epoch_max=epochs)
	print(previous_model)
	if os.path.exists(os.path.join(experdir, 'model-final.hdf5')):
		print("Already trained model, skipping.")
		return None, None
	else:
		if multi_gpu > 0:
			strategy = tf.distribute.MirroredStrategy()
			with strategy.scope():
				model.build(input_shape=input_shape, optimizer=optimfun, margin=margin,
							softmax_attention=softmax_attention, dropout=dropout, weight_decay=weight_decay,
							attention_drop_rate=attention_drop_rate, crossentropy_weight=cross_weight,
							nclasses=nclasses, norm_triplet_loss=norm_triplet_loss,
							soft_triplet_loss=soft_triplet_loss, kernel_regularizer=kernel_regularizer,
							split_crossentropy=split_crossentropy,
							shared_weights_crossentropy=shared_weights_crossentropy, n_filters=n_filters,
							adaptative_loss=adaptative_triplet_loss, reduction=reduction,
							combined_output_length=combined_output_length)
		else:
			model.build(input_shape=input_shape, optimizer=optimfun, margin=margin,
						softmax_attention=softmax_attention, dropout=dropout, weight_decay=weight_decay,
						attention_drop_rate=attention_drop_rate, crossentropy_weight=cross_weight,
						nclasses=nclasses, norm_triplet_loss=norm_triplet_loss,
						soft_triplet_loss=soft_triplet_loss, kernel_regularizer=kernel_regularizer,
						split_crossentropy=split_crossentropy,
						shared_weights_crossentropy=shared_weights_crossentropy, n_filters=n_filters,
						adaptative_loss=adaptative_triplet_loss, reduction=reduction,
						combined_output_length=combined_output_length)


		model.model.summary()

		if init_net != "":
			model.model.load_weights(init_net, by_name=True, skip_mismatch=True)
			model.model_encode.load_weights(init_net, by_name=True, skip_mismatch=True)
			print("INFO: Weights loaded from ", init_net, flush=True)

		pattern_file = "model-state-{:04d}_weights.hdf5"
		previous_model = find_latest_file_model(experdir, pattern_file, epoch_max=epochs + extra_epochs)
		print(previous_model)
		if previous_model != "":
			pms = previous_model.split("-")
			initepoch = int(pms[len(pms) - 1].split("_")[0])
			print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
			model.model.load_weights(previous_model, by_name=True, skip_mismatch=True)
			model.model_encode.load_weights(previous_model, by_name=True, skip_mismatch=True)
			iters = train_generator.__len__() * initepoch
		else:
			initepoch = 0
			iters = 0

		# ---------------------------------------
		# Train model
		# --------------------------------------
		if verbose > 1:
			print(experdir)

		if multi_gpu:
			options = tf.data.Options()
			options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
			train_generator.dataset = train_generator.dataset.with_options(options)
			train_generator.dataset = strategy.experimental_distribute_dataset(train_generator.dataset)

		with strategy.scope():
			def compute_loss(labels, predictions, global_batch_size, weights):
				losses = []
				if len(model.losses) > 1:
					for loss_ix in range(len(model.losses)):
						loss = model.losses[loss_ix](list(labels.values())[loss_ix], predictions[loss_ix])
						loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
						losses.append(weights[loss_ix] * loss)
				else:
					loss = model.losses[0](list(labels.values())[0], predictions)
					loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
					losses.append(weights[0] * loss)

				if model.model.losses:
					losses.append(tf.nn.scale_regularization_loss(tf.add_n(model.model.losses)))

				return losses

		# @tf.function
		def train_step(x, labels, global_batch_size, weights):
			with tf.GradientTape() as tape:
				predictions = model.model(x, training=True)
				losses = compute_loss(labels, predictions, global_batch_size, weights)
				loss_value = sum(losses)
				if use_mixed_precision:
					loss_value = optimfun.get_scaled_loss(loss_value)
			grads = tape.gradient(loss_value, model.model.trainable_weights)
			if use_mixed_precision:
				grads = optimfun.get_unscaled_gradients(grads)
			optimfun.apply_gradients(zip(grads, model.model.trainable_weights))
			return loss_value, losses

		@tf.function
		def distributed_train_step(x, labels, global_batch_size, weights):
			per_replica_loss, per_replica_losses = strategy.run(train_step,
			                                                    args=(x, labels, global_batch_size, weights,))
			return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None), per_replica_losses

		# ---------------------------------------
		# Train loop
		# ---------------------------------------
		iterator = iter(train_generator.dataset)
		extra_epochs_done = False

		for epoch in range(initepoch, epochs + extra_epochs):
			epoch_losses = np.zeros(len(model.losses))
			total_losses = 0
			print("\nStart of epoch %d" % (epoch,))
			start_time = time.time()

			for step in range(train_generator.__len__()):
				if epoch == 3 and step == 10:
					tf.profiler.experimental.start(experdir_tensorboard)
				if epoch == 3 and step == 60:
					tf.profiler.experimental.stop()

				x_batch_train, y_batch_train = next(iterator)
				if multi_gpu > 0:
					loss_value, per_replica_losses = distributed_train_step(x_batch_train, y_batch_train, batchsize,
					                                                        model.losses_weights)
					if len(model.losses) > 1:
						for loss_ix in range(len(model.losses)):
							epoch_losses[loss_ix] = epoch_losses[loss_ix] + strategy.reduce(tf.distribute.ReduceOp.SUM,
							                                                                per_replica_losses[loss_ix],
							                                                                axis=None)
					else:
						epoch_losses[0] = loss_value
				else:
					loss_value, losses = train_step(x_batch_train, y_batch_train, batchsize, model.losses_weights)
					if len(model.losses) > 1:
						epoch_losses = epoch_losses + np.asarray(losses)

				total_losses = total_losses + loss_value
				iters = iters + 1

				if step % 10 == 0:
					if len(epoch_losses) > 1:
						epoch_losses_ = np.insert(epoch_losses, 0, total_losses)
					else:
						epoch_losses_ = epoch_losses

					if lr_sched:
						print("Training losses (for one batch) at step %d" % step + ": [" + ", ".join(
							['%.6f'] * len(epoch_losses_)) % tuple(
							epoch_losses_ / (step + 1)) + "], lr: %.4f" % lr_.__call__(iters))
					else:
						print("Training losses (for one batch) at step %d" % step + ": [" + ", ".join(
							['%.6f'] * len(epoch_losses_)) % tuple(epoch_losses_ / (step + 1)) + "], lr: %.4f" % lr_)
					print("Seen so far: %s samples" % ((step + 1) * batchsize), flush=True)

			epoch_losses = epoch_losses / train_generator.__len__()
			if len(epoch_losses) > 1:
				epoch_losses_ = np.insert(epoch_losses, 0, total_losses)
			else:
				epoch_losses_ = epoch_losses


			print("Epoch loss: %.2f " % epoch_losses_[0] + "[" + ", ".join(['%.4f'] * len(epoch_losses_)) % tuple(
				epoch_losses_) + "]" + "- Time taken: %.2fs" % (time.time() - start_time))

			with train_summary_writer.as_default():
				tf.summary.scalar('loss', epoch_losses[0], step=epoch)

				tf.summary.scalar('triplet_loss', epoch_losses[1], step=epoch)

				if crossentropy_weight > 0.0:
					tf.summary.scalar('crossentropy_loss', epoch_losses[-1], step=epoch)

				if lr_sched:
					tf.summary.scalar("learning_rate", lr_.__call__(iters), step=epoch)
				else:
					tf.summary.scalar("learning_rate", lr_, step=epoch)

			if epoch % save_epoch == 0:
				model.save(epoch)

			if extra_epochs > 0 and epoch >= epochs and extra_epochs_done == False:
				print("Starting training without crossentropy")
				model.losses_weights[-1] = 0.0
				extra_epochs_done = True
				if not lr_sched:
					lr_ = lr_ * 0.1

		return model, experdir


################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Trains a CNN for gait')
	parser.add_argument('--debug', default=False, action='store_true')
	parser.add_argument('--lr', type=float, required=False,
						default=0.0001,
						help='Starting learning rate')
	parser.add_argument('--datadir', type=str, required=True,
						help="Full path to data directory")
	parser.add_argument('--experdir', type=str, required=True,
						help="Base path to save results of training")
	parser.add_argument('--prefix', type=str, required=True,
						default="demo",
						help="String to prefix experiment directory name.")
	parser.add_argument('--bs', type=int, required=False,
						default=64,
						help='Batch size')
	parser.add_argument('--epochs', type=int, required=False,
						default=75,
						help='Maximum number of epochs')
	parser.add_argument('--extraepochs', type=int, required=False,
						default=25,
						help='Extra number of epochs to add validation data')
	parser.add_argument('--nclasses', type=int, required=True,
						default=74,
						help='Maximum number of epochs')
	parser.add_argument('--optimizer', type=str, required=False,
						default="Adam",
						help="Optimizer: SGD, Adam, AMSGrad")
	parser.add_argument("--verbose", type=int,
						nargs='?', const=False, default=1,
						help="Whether to enable verbosity of output")
	parser.add_argument('--margin', type=float, required=False,
						default=0.2,
						help='Margin for triplet loss')
	parser.add_argument('--pk', required=False, default=False, action='store_true', help='Use p*q?')
	parser.add_argument('--p', type=int, required=False, default=8, help='Number of classes in the batch')
	parser.add_argument('--k', type=int, required=False, default=16, help='Number of samples per class')
	parser.add_argument('--wd', type=float, required=False, default=0.00001, help='Weight decay')
	parser.add_argument('--cut', required=False, default=False, action='store_true', help='Remove background?')
	parser.add_argument('--diffFrames', required=False, default=False, action='store_true', help='Different frames per batch?')
	parser.add_argument('--random_frames', required=False, default=False, action='store_true',help='Disordered frames per batch?')
	parser.add_argument('--save_epoch', type=int, required=False, default=100, help='Save epoch')
	parser.add_argument('--augmentation', required=False, default=False, action='store_true', help='Data augmentation?')
	parser.add_argument('--dropout', type=float, required=False, default=0, help='Dropout rate')
	parser.add_argument('--attention_drop_rate', type=float, required=False, default=0.25, help='Attention dropout rate')
	parser.add_argument('--cross_weight', type=float, required=False, default=0, help='Crossentropy loss weight')
	parser.add_argument('--norm_triplet_loss', default=False, action='store_true')
	parser.add_argument('--soft_triplet_loss', default=False, action='store_true')
	parser.add_argument('--kernel_regularizer', default=False, action='store_true')
	parser.add_argument('--lr_sched', default=False, action='store_true')
	parser.add_argument('--split_crossentropy', default=False, action='store_true')
	parser.add_argument('--shared_weights_crossentropy', default=False, action='store_true')
	parser.add_argument('--combine_outputs', default=False, action='store_true')
	parser.add_argument('--softmax_attention', default=False, action='store_true')
	parser.add_argument('--init_net', type=str, required=False, default='', help='Path to initial model')
	parser.add_argument('--substract_mean', default=False, action='store_true')
	parser.add_argument('--multi_gpu', type=int, required=False, default=0)
	parser.add_argument('--adaptative_triplet_loss', default=False, action='store_true')
	parser.add_argument('--no_clip', default=False, action='store_true')
	parser.add_argument('--reduction', type=str, required=False,default="both", help="both, max.")
	parser.add_argument('--model_size', type=str, required=False, default="small", help="small, medium, large.")
	parser.add_argument('--combined_output_length', type=int, required=False, default=16, help='Feture length per part for the combined output')
	parser.add_argument('--repeat_data', default=False, action='store_true')
	parser.add_argument('--self_attention', default=False, action='store_true')
	parser.add_argument('--mixed_precision', default=False, action='store_true')

	args = parser.parse_args()
	verbose = args.verbose
	datadir = args.datadir
	prefix = args.prefix
	epochs = args.epochs
	extraepochs = args.extraepochs
	batchsize = args.bs
	nclasses = args.nclasses
	lr = args.lr
	optimizer = args.optimizer
	experdirbase = args.experdir
	IS_DEBUG = args.debug
	margin = args.margin
	pk = args.pk
	p = args.p
	k = args.k
	weight_decay = args.wd
	cut = args.cut
	diffFrames = args.diffFrames
	save_epoch = args.save_epoch
	augmentation = args.augmentation
	dropout = args.dropout
	attention_drop_rate = args.attention_drop_rate
	cross_weight = args.cross_weight
	norm_triplet_loss = args.norm_triplet_loss
	soft_triplet_loss = args.soft_triplet_loss
	kernel_regularizer = args.kernel_regularizer
	lr_sched = args.lr_sched
	split_crossentropy = args.split_crossentropy
	shared_weights_crossentropy = args.shared_weights_crossentropy
	combine_outputs = args.combine_outputs
	model_size = args.model_size
	init_net = args.init_net
	softmax_attention = args.softmax_attention
	random_frames = args.random_frames
	multi_gpu = args.multi_gpu
	adaptative_triplet_loss = args.adaptative_triplet_loss
	substract_mean = args.substract_mean
	reduction = args.reduction
	combined_output_length = args.combined_output_length
	repeat_data = args.repeat_data
	use_mixed_precision = args.mixed_precision

	if use_mixed_precision:
		from tensorflow.keras import mixed_precision
		mixed_precision.set_global_policy('mixed_float16')

	# Start the processing
	if nclasses == 50:
		# Train as many models as cameras.
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

		for camera in cameras:
			cameras_ = cameras.copy()
			cameras_.remove(camera)
			print("Fine tuning with ", cameras_, " cameras")
			final_model, experdir = trainGaitNet(datadir=datadir, experfix=prefix, lr=lr,
												 experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
												 epochs=epochs, batchsize=batchsize,
												 extra_epochs=extraepochs, verbose=verbose, margin=margin, pk=pk, p=p,
												 k=k,
												 weight_decay=weight_decay, cut=cut, diffFrames=diffFrames,
												 save_epoch=save_epoch, augmentation=augmentation, dropout=dropout,
												 attention_drop_rate=attention_drop_rate,
												 crossentropy_weight=cross_weight,
												 norm_triplet_loss=norm_triplet_loss,
												 soft_triplet_loss=soft_triplet_loss,
												 lr_sched=lr_sched, kernel_regularizer=kernel_regularizer,
												 split_crossentropy=split_crossentropy,
												 shared_weights_crossentropy=shared_weights_crossentropy,
												 combine_outputs=combine_outputs, model_size=model_size,
												 init_net=init_net, softmax_attention=softmax_attention,
												 random_frames=random_frames, multi_gpu=multi_gpu,
												 adaptative_triplet_loss=adaptative_triplet_loss,
												 substract_mean=substract_mean,
												 reduction=reduction, combined_output_length=combined_output_length,
												 repeat_data=repeat_data)

			if final_model is not None:
				final_model.save(os.path.join(experdir, "model-final.hdf5"), include_optimizer=False)
	else:
		final_model, experdir = trainGaitNet(datadir=datadir, experfix=prefix, lr=lr,
											 experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
											 epochs=epochs, batchsize=batchsize,
											 extra_epochs=extraepochs, verbose=verbose, margin=margin, pk=pk, p=p, k=k,
											 weight_decay=weight_decay, cut=cut, diffFrames=diffFrames,
											 save_epoch=save_epoch, augmentation=augmentation,  dropout=dropout,
											 attention_drop_rate=attention_drop_rate, crossentropy_weight=cross_weight,
											 norm_triplet_loss=norm_triplet_loss, soft_triplet_loss=soft_triplet_loss,
											 lr_sched=lr_sched, kernel_regularizer=kernel_regularizer,
											 split_crossentropy=split_crossentropy,
											 shared_weights_crossentropy=shared_weights_crossentropy,
											 combine_outputs=combine_outputs, model_size=model_size,
		                                     init_net=init_net, softmax_attention=softmax_attention,
		                                     random_frames=random_frames, multi_gpu=multi_gpu,
		                                     adaptative_triplet_loss=adaptative_triplet_loss, substract_mean=substract_mean,
		                                     reduction=reduction, combined_output_length=combined_output_length,
		                                     repeat_data=repeat_data)

		if final_model is not None:
			final_model.model.save(os.path.join(experdir, "model-final.hdf5"), include_optimizer=False)

	print("* End of training: {}".format(experdir))

