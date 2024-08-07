import os
import sys
import numpy as np
import os.path as osp
import pathlib

np.object = object    # Fix for deepdish

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")

import deepdish as dd
from sklearn.metrics import confusion_matrix
from data.data_generator import DataGeneratorGait
from misc.knn import KNN

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
	all_cams = []
	all_gaits = []
	nbatches = data_generator.__len__()
	for bix in range(nbatches):
		print("Encoding ", bix, "/", nbatches, end='\r')
		data, labels, videoId, cams = data_generator.__getitemvideoid__(bix)

		feats = model.encode(data, batch_size=1)
		feats = feats[0]
		all_feats.append(feats)
		all_vids.append(videoId)
		all_gt_labs.append(labels)
		all_cams.append(cams)
		pos = np.where(data_generator.videoId == videoId)[0]
		all_gaits.append([data_generator.gait[pos[0]]])

	return all_feats, all_gt_labs, all_vids, all_cams, all_gaits

def test(gallery_feats, gallery_gt_labs, test_feats, test_gt_labs, clf, metric, k, p=None):
	pred_labs = clf.predict(gallery_feats, gallery_gt_labs, test_feats, metric, k, p)

	M = confusion_matrix(test_gt_labs, pred_labs)
	acc = M.diagonal().sum() / len(test_gt_labs)

	return acc

def encode_test(model, datadir, nclasses=50, batchsize=128,
                nframes=None, cut=False):
	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	if nclasses == 50:
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		data_folder_probe = osp.join(datadir, 'tfimdb_casiab_N050_test_of30_64x64')
		info_file_probe = osp.join(datadir, 'tfimdb_casiab_N050_test_of30_64x64.h5')
		dataset_info_probe = dd.io.load(info_file_probe)

		testdir = os.path.join(experdir, "results")
		os.makedirs(testdir, exist_ok=True)
		outpath = os.path.join(testdir, "probe_{:03}_{:02}_knn.h5".format(nclasses, nframes))

		if not os.path.exists(outpath):
			probe_generator = DataGeneratorGait(dataset_info_probe, batch_size=batchsize, mode='test',
												  labmap=None, camera=cameras,
												  datadir=data_folder_probe, augmentation=False, max_frames=nframes,
												  cut=cut)
			all_feats_probe, all_gt_labs_probe, all_vids_probe, all_cams_probe, all_gaits_probe = encodeData(probe_generator, model)

			all_feats_probe = np.concatenate(np.expand_dims(all_feats_probe, 1), axis=0)
			all_gt_labs_probe = np.asarray(np.concatenate(all_gt_labs_probe))
			all_vids_probe = np.asarray(np.concatenate(all_vids_probe))
			all_cams_probe = np.asarray(np.concatenate(all_cams_probe))
			all_gaits_probe = np.asarray(np.concatenate(all_gaits_probe))

			# Save CM
			exper = {}
			exper["feats"] = all_feats_probe
			exper["gtlabs"] = all_gt_labs_probe
			exper["vids"] = all_vids_probe
			exper["cams"] = all_cams_probe
			exper["gaits"] = all_gaits_probe
			print("Saving data:")
			dd.io.save(outpath, exper)
			print("Data saved to: " + outpath)
		else:
			exper = dd.io.load(outpath)
			all_feats_probe = exper["feats"]
			all_gt_labs_probe = exper["gtlabs"]
			all_vids_probe = exper["vids"]
			all_cams_probe = exper["cams"]
			all_gaits_probe = exper["gaits"]
	else:
		sys.exit(0)

	return all_feats_probe, all_gt_labs_probe, all_vids_probe, all_cams_probe, all_gaits_probe

def encode_gallery(model, datadir, nclasses=50, batchsize=128, nframes=None, cut=False):
	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	if nclasses == 50:
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		data_folder_gallery = osp.join(datadir, 'tfimdb_casiab_N050_ft_of30_64x64')
		info_file_gallery = osp.join(datadir, 'tfimdb_casiab_N050_ft_of30_64x64.h5')
		dataset_info_gallery = dd.io.load(info_file_gallery)

		testdir = os.path.join(experdir, "results")
		os.makedirs(testdir, exist_ok=True)
		outpath = os.path.join(testdir, "gallery_{:03}_{:02}_knn.h5".format(nclasses, nframes))

		if not os.path.exists(outpath):
			gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
			                                      labmap=None, camera=cameras,
			                                      datadir=data_folder_gallery, augmentation=False, max_frames=nframes,cut=cut)
			all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery, all_gaits_gallery = encodeData(gallery_generator, model)

			all_feats_gallery = np.concatenate(np.expand_dims(all_feats_gallery, 1), axis=0)
			all_gt_labs_gallery = np.asarray(np.concatenate(all_gt_labs_gallery))
			all_vids_gallery = np.asarray(np.concatenate(all_vids_gallery))
			all_cams_gallery = np.asarray(np.concatenate(all_cams_gallery))
			all_gaits_gallery = np.asarray(np.concatenate(all_gaits_gallery))

			# Save CM
			exper = {}
			exper["feats"] = all_feats_gallery
			exper["gtlabs"] = all_gt_labs_gallery
			exper["vids"] = all_vids_gallery
			exper["cams"] = all_cams_gallery
			exper["gaits"] = all_gaits_gallery
			print("Saving data:")
			dd.io.save(outpath, exper)
			print("Data saved to: " + outpath)
		else:
			exper = dd.io.load(outpath)
			all_feats_gallery = exper["feats"]
			all_gt_labs_gallery = exper["gtlabs"]
			all_vids_gallery = exper["vids"]
			all_cams_gallery = exper["cams"]
			all_gaits_gallery = exper["gaits"]
	else:
		sys.exit(0)

	return all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery, all_gaits_gallery

################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

	parser.add_argument('--allcameras', default=False, action='store_true',
	                    help="Test with all cameras")

	parser.add_argument('--datadir', type=str, required=False,
						help="Full path to data directory")

	parser.add_argument('--model', type=str, required=True,
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

	parser.add_argument('--camera', type=int, required=False,
	                    default=90,
	                    help='Camera')

	parser.add_argument('--metrics', type=str, required=False,
	                    default="L2",
	                    help="gray|depth|of|rgb")

	parser.add_argument('--nframes', type=int, required=False,
						default=30,
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

	args = parser.parse_args()
	datadir = args.datadir
	batchsize = args.bs
	nclasses = args.nclasses
	modelpath = args.model
	camera = args.camera
	allcameras = args.allcameras
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

	# Call the evaluator
	if allcameras:
		test_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		gallery_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		conds = 3
	else:
		test_cameras = [camera]
		gallery_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		conds = 3

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
	nclasses_train = 74

	model.build(input_shape=input_shape, optimizer=optimfun, margin=margin, nclasses=nclasses_train,
	            crossentropy_weight=cross_weight, n_filters=n_filters, split_crossentropy=split_crossentropy, softmax_attention=softmax_attention,
	            combined_output_length=combined_output_length)

	model.model.summary()
	model.model.load_weights(modelpath, by_name=True, skip_mismatch=True)
	model.model_encode = tf.keras.Model(model.model.input, model.model.get_layer(encode_layer).output)

	# ---------------------------------------
	# Compute ACCs
	# ---------------------------------------
	gallery_feats, gallery_gt_labs, gallery_vids, gallery_cams, gallery_gaits = encode_gallery(model, datadir, nclasses,
																				batchsize, nframes, cut=cut)
	test_feats, test_gt_labs, test_vids, test_cams, test_gaits = encode_test(model, datadir, nclasses,
																 batchsize, nframes, cut=cut)
	clf = KNN(gpu=0)
	accs_global = np.zeros((conds, len(test_cameras)+1))
	for test_cam_ix in range(len(test_cameras)):
		test_cam = test_cameras[test_cam_ix]
		for gait_cond_ix in range(conds):
			acc_ = 0
			acc_video = 0
			for gallery_cam_ix in range(len(gallery_cameras)):
				gallery_cam = gallery_cameras[gallery_cam_ix]
				if test_cam != gallery_cam:
					pos = np.where(np.asarray(gallery_cams) == gallery_cam)[0]
					gallery_feats_ = gallery_feats[pos, :]
					gallery_gt_labs_ = gallery_gt_labs[pos]
					pos = np.where((np.asarray(test_cams) == test_cam) & (np.asarray(test_gaits) == gait_cond_ix))[0]
					test_feats_ = test_feats[pos, :]
					test_gt_labs_ = test_gt_labs[pos]
					acc = test(gallery_feats_, gallery_gt_labs_, test_feats_, test_gt_labs_, clf, metrics, knn)
					acc_ = acc_ + acc

			acc_ = acc_ / (len(gallery_cameras) - 1)
			accs_global[gait_cond_ix, test_cam_ix] = acc_

	# Compute the avg per walking pattern
	accs_global[:,-1] = np.sum(accs_global[:, 0:-1], axis=1) / len(test_cameras)
	print("Done!")
	print("acc:")
	print("nm:", accs_global[0, :])
	print("bg:", accs_global[1, :])
	print("cl:", accs_global[2, :])
