import cv2
import numpy as np
import argparse
import os
import deepdish as dd
import glob

np.object = object    # Fix for deepdish

def loadVideo(video_path):
	frame_list = sorted(glob.glob(os.path.join(video_path, '*.png')))
	if len(frame_list) > 0:
		frames = []
		for frame_file in frame_list:
			# Load frame-by-frame
			frame = cv2.imread(frame_file)
			frame = np.expand_dims(frame, axis=0)
			frames.append(frame)

		return np.vstack(frames)
	else:
		return None

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build OF train dataset.')

parser.add_argument('--ofdir', type=str, required=False,
                    default='GREW_pre/flow/train',
					help='Full path to of directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='GREW_tf_pre/',
                    help="Full path for output files. Note that one or more folders are created to store de files")

args = parser.parse_args()

ofdir = args.ofdir
outdir = args.outdir

# Initialize some parameters...
np.random.seed(0)

videoId = 1
labels_ = []
videoIds_ = []
gaits_ = []
frames_ = []
bbs_ = []
file_ = []
meanSample = np.zeros((64,64,3), dtype=np.float32)

# Subjects loop
id_folders = glob.glob(os.path.join(ofdir, '*/'))
folder_dir = 'tfimdb_grew_N20000_train_of' + str(25) + '_64x64'
os.makedirs(os.path.join(outdir, folder_dir), exist_ok=True)

for id_ix in id_folders:
	# Video loop
	video_folders = glob.glob(os.path.join(id_ix, '*/'))
	for video_ix in video_folders:
		# Load files.
		im = loadVideo(video_ix)

		if im is not None:
			# Stack n_frames continuous frames
			id = int(id_ix.split('/')[-2])

			pattern = video_ix.split('/')[-2]

			outpath = os.path.join(outdir, folder_dir, "{:03d}".format(id) + '_' + pattern + '.h5')
			if not os.path.exists(outpath):
				positions = list(range(im.shape[0]))
				# Write output file.
				data = dict()
				data['data'] = np.uint8(im)
				data['label'] = np.uint16(id)
				data['videoId'] = np.uint16(videoId)
				data['gait'] = np.uint8(0)
				data['frames'] = np.uint16(positions)
				data['bbs'] = np.uint8([])
				data['compressFactor'] = np.uint8(0)
				meanSample = meanSample + np.sum(im, axis=0)
				dd.io.save(outpath, data)
			else:
				data = dd.io.load(outpath)
				sub_position_list = data['bbs']
				positions = data['frames']

			# Append data for the global file
			labels_.append(id)
			videoIds_.append(videoId)
			gaits_.append(0)
			bbs_.append([])
			frames_.append(positions)
			file_.append("{:03d}".format(id) + '_' + pattern + '.h5')

			videoId = videoId + 1

# Get train/val/test sets.
set_ = np.zeros(len(labels_))
np.random.seed(0)
labels = np.uint16(np.asarray(labels_))
gg = np.uint8(np.asarray(gaits_))
ulabs = np.unique(labels)
ugaits = np.unique(gg)
nval_samples = len(labels_)
nsamples_per_id_gait = int(nval_samples / (len(ulabs) * len(ugaits)))
for id in ulabs:
	for gait in ugaits:
		pos_lab = np.where(labels == id)[0]
		pos_gait = np.where(gg == gait)[0]
		common_pos = np.intersect1d(pos_lab, pos_gait)
		np.random.shuffle(common_pos)
		en_pos = len(common_pos) - nsamples_per_id_gait
		train_samples = common_pos[0:en_pos]
		val_samples = common_pos[en_pos:len(common_pos)]
		set_[train_samples] = 1
		set_[val_samples] = 2

assert np.count_nonzero(set_) == len(set_)

# Write global file
data = dict()
data['label'] = np.uint16(np.asarray(labels_))
data['videoId'] = np.uint16(np.asarray(videoIds_))
data['gait'] = np.uint8(np.asarray(gaits_))
data['set'] = np.uint8(set_)
data['frames'] = frames_
data['bbs'] = bbs_
data['compressFactor'] = np.uint8(0)
data['file'] = file_
data['shape'] = (60, 60, 25)
data['mean'] = meanSample / np.float32(len(labels_))
outpath = os.path.join(outdir, folder_dir + '.h5')
dd.io.save(outpath, data)
