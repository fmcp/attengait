import os
import sys
import numpy as np
import os.path as osp
from os.path import expanduser

import pathlib

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")
homedir = expanduser("~")

import deepdish as dd
from data.data_generator import DataGeneratorGait
from time import strftime, localtime

# --------------------------------
import tensorflow as tf

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate # TODO
tf.config.run_functions_eagerly(True)
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()


# --------------------------------
def encodeData(data_generator, model):
	all_vids = []
	all_gt_labs = []
	all_feats = []
	all_files = []
	nbatches = data_generator.__len__()
	for bix in range(nbatches):
		print("Encoding ", bix, "/", nbatches, end='\r')
		data, labels, videoId, cams = data_generator.__getitemvideoid__(bix)
		feats = model.encode(data, batch_size=1)
		feats = feats[0]
		all_feats.append(feats)
		all_vids.append(videoId)
		all_gt_labs.append(labels)
		all_files.append(cams[0].decode('UTF-8'))

	return all_feats, all_gt_labs, all_vids, all_files

def encode_test(model, datadir, nclasses=50, batchsize=128,
                nframes=None, sufix="", cut=False):

	all_feats_global = []
	all_gt_labs_global = []
	all_vids_global = []

	data_folder = osp.join(datadir, 'tfimdb_grew_N06000_test_of30_64x64')
	info_file = osp.join(datadir, 'tfimdb_grew_N06000_test_of30_64x64.h5')
	dataset_info = dd.io.load(info_file)

	testdir = os.path.join(experdir, "results")
	outpath = os.path.join(testdir, "test_{:05}_{:02}_{}.h5".format(nclasses, nframes, sufix))

	if not os.path.exists(outpath):
		test_generator = DataGeneratorGait(dataset_info, batch_size=batchsize, mode='test', labmap=None,
		                                   datadir=data_folder, max_frames=nframes, cut=cut)

		all_feats, all_gt_labs, all_vids, all_files = encodeData(test_generator, model)

		# Save CM
		exper = {}
		exper["feats"] = np.concatenate(np.expand_dims(all_feats, 1), axis=0)
		exper["gtlabs"] = np.asarray(all_gt_labs)
		exper["vids"] = np.asarray(all_vids)
		exper["files"] = all_files

		if outpath is not None:
			dd.io.save(outpath, exper)
			print("Data saved to: " + outpath)
	else:
		exper = dd.io.load(outpath)
		all_feats = exper["feats"]
		all_gt_labs = exper["gtlabs"]
		all_vids = exper["vids"]
		all_files = exper["files"]

	all_feats_global.append(all_feats)
	all_gt_labs_global.append(all_gt_labs)
	all_vids_global.append(all_vids)
	all_files_global = all_files

	for i in range(len(all_feats_global)):
		all_feats_global[i] = np.asarray(all_feats_global[i])
		all_gt_labs_global[i] = np.asarray(all_gt_labs_global[i])
		all_vids_global[i] = np.asarray(all_vids_global[i])

	return all_feats_global, all_gt_labs_global, all_vids_global, all_files_global

def encode_gallery(model, datadir, nclasses=50, batchsize=128, nframes=None,
				   cut=False):
	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	data_folder_gallery = osp.join(datadir, 'tfimdb_grew_N06000_ft_of30_64x64')
	info_file_gallery = osp.join(datadir, 'tfimdb_grew_N06000_ft_of30_64x64.h5')
	dataset_info_gallery = dd.io.load(info_file_gallery)

	testdir = os.path.join(experdir, "results")
	os.makedirs(testdir, exist_ok=True)
	outpath = os.path.join(testdir, "gallery_{:05}_{:02}_knn.h5".format(nclasses, nframes))

	if not os.path.exists(outpath):
		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
		                                      labmap=None, camera=None,
		                                      datadir=data_folder_gallery, augmentation=False,
		                                      max_frames=nframes, cut=cut)
		all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_files_gallery = encodeData(gallery_generator, model)

		all_feats_gallery = np.concatenate(np.expand_dims(all_feats_gallery, 1), axis=0)
		all_gt_labs_gallery = np.asarray(np.concatenate(all_gt_labs_gallery))
		all_vids_gallery = np.asarray(np.concatenate(all_vids_gallery))

		# Save CM
		exper = {}
		exper["feats"] = all_feats_gallery
		exper["gtlabs"] = all_gt_labs_gallery
		exper["vids"] = all_vids_gallery
		print("Saving data:")
		dd.io.save(outpath, exper)
		print("Data saved to: " + outpath)
	else:
		exper = dd.io.load(outpath)
		all_feats_gallery = exper["feats"]
		all_gt_labs_gallery = exper["gtlabs"]
		all_vids_gallery = exper["vids"]

	return all_feats_gallery, all_gt_labs_gallery, all_vids_gallery

################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

	parser.add_argument('--datadir', type=str, required=False,
						default=osp.join('/home/GAIT_local/SSD', 'TUM_GAID_tf'),
						help="Full path to data directory")

	parser.add_argument('--model', type=str, required=True,
						default=osp.join(homedir,
										 'experiments/tumgaid_mj_tf/tum150gray_datagen_opAdam_bs128_lr0.001000_dr0.30/model-state-0002.hdf5'),
						help="Full path to model file (DD: .hdf5)")

	parser.add_argument('--bs', type=int, required=False,
						default=128,
						help='Batch size')

	parser.add_argument('--nclasses', type=int, required=True,
						default=155,
						help='Maximum number of epochs')

	parser.add_argument('--knn', type=int, required=True,
	                    default=7,
	                    help='Number of noighbours')

	parser.add_argument('--metrics', type=str, required=False,
	                    default="L2",
	                    help="gray|depth|of|rgb")

	parser.add_argument('--nframes', type=int, required=False,
						default=25,
						help='Number Frames')

	parser.add_argument('--cut', required=False, default=False, action='store_true', help='Remove background?')

	parser.add_argument('--verbose', default=False, action='store_true', help="Verbose")
	parser.add_argument('--cross_weight', type=float, required=False, default=0, help='Crossentropy loss weight')
	parser.add_argument('--model_size', type=str, required=False, default='small', help='Model size: small, medium, large')
	parser.add_argument('--split_crossentropy', default=False, action='store_true')
	parser.add_argument('--softmax_attention', default=False, action='store_true')
	parser.add_argument('--combined_output_length', type=int, required=False,
	                    default=32,
	                    help='Outputs length per part')
	parser.add_argument('--no_clip', default=False, action='store_true')

	args = parser.parse_args()
	datadir = args.datadir
	batchsize = args.bs
	nclasses = args.nclasses
	modelpath = args.model
	lstm = args.lstm
	knn = args.knn
	metrics = args.metrics
	metrics = [metrics]
	nframes = args.nframes
	cut = args.cut
	verbose = args.verbose
	cross_weight = args.cross_weight
	model_size = args.model_size
	split_crossentropy = args.split_crossentropy
	softmax_attention = args.softmax_attention
	combined_output_length = args.combined_output_length

	from misc.knn import KNN

	# ---------------------------------------
	# Load model
	# ---------------------------------------
	experdir, filename = os.path.split(modelpath)

	from nets.attengait import GaitSetTransformer as OurModel
	encode_layer = "encode"
	model = OurModel(experdir, reduce_channel=False, batch_size=batchsize)
	input_shape = (None, 64, 64, 3)

	if model_size == "small":
		n_filters = [32, 64, 128, 256]
	elif model_size == "medium":
		n_filters = [64, 128, 256, 256]
	elif model_size == "large":
		n_filters = [128, 256, 512, 256]
	else:
		n_filters = [32, 64, 128, 256]

	print(modelpath)
	optimfun = tf.keras.optimizers.SGD()
	margin = 0.2
	nclasses_train = 20000

	model.build(input_shape=input_shape, optimizer=optimfun, margin=margin, nclasses=nclasses_train,
	            crossentropy_weight=cross_weight, n_filters=n_filters, split_crossentropy=split_crossentropy, softmax_attention=softmax_attention,
	            combined_output_length=combined_output_length)

	model.model.summary()
	model.model.load_weights(modelpath, by_name=True, skip_mismatch=True)
	model.model_encode = tf.keras.Model(model.model.input, model.model.get_layer(encode_layer).output)

	# ---------------------------------------
	# Compute ACCs
	# ---------------------------------------
	gallery_feats, gallery_gt_labs, gallery_vids = encode_gallery(model, datadir, nclasses, batchsize, nframes,
	                                                              cut=cut)

	test_feats, test_gt_labs, test_vids, test_files = encode_test(model, datadir, nclasses, batchsize, nframes, cut=cut)

	clf = KNN(gpu=0)
	pred_labs = clf.predict(gallery_feats, gallery_gt_labs, test_feats[0], metrics, 20, None)
	save_path = os.path.join(experdir, "results/" + strftime('%Y-%m%d-%H%M%S', localtime()) + ".csv")
	with open(save_path, "w") as f:
		f.write(
			"videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
		for i in range(len(pred_labs)):
			r_format = [int(idx) for idx in pred_labs[i]]
			output_row = '{}' + ',{}' * 20 + '\n'
			f.write(output_row.format(test_files[i][:-3], *r_format))
		print("GREW result saved to {}/{}".format(os.getcwd(), save_path))