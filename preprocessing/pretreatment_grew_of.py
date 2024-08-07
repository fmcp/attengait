import os
import cv2
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

def boolean_string(s):
	if s.upper() not in {'FALSE', 'TRUE'}:
		raise ValueError('Not a valid boolean string')
	return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
					help='Root path of raw of dataset.')
parser.add_argument('--output_path', default='', type=str,
					help='Root path for output.')
parser.add_argument('--mode', default='train', type=str,
					help='train/test')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
					help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
					help='If set as True, all logs will be saved. '
						 'Otherwise, only warnings and errors will be saved.'
						 'Default: False')
parser.add_argument('--worker_num', default=1, type=int,
					help='How many subprocesses to use for data pretreatment. '
						 'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num
MODE = opt.mode
T_H = 64
T_W = 64


def log2str(pid, comment, logs):
	str_log = ''
	if type(logs) is str:
		logs = [logs]
	for log in logs:
		str_log += "# JOB %d : --%s-- %s\n" % (
			pid, comment, log)
	return str_log


def log_print(pid, comment, logs):
	str_log = log2str(pid, comment, logs)
	if comment in [WARNING, FAIL]:
		with open(LOG_PATH, 'a') as log_f:
			log_f.write(str_log)
	if comment in [START, FINISH]:
		if pid % 500 != 0:
			return
	print(str_log, end='')


def cut_pickle(seq_info, pid):
	seq_name = '-'.join(seq_info)
	log_print(pid, START, seq_name)
	seq_path = os.path.join(INPUT_PATH, *seq_info)
	out_dir = os.path.join(OUTPUT_PATH, *seq_info)
	frame_list = os.listdir(seq_path)
	frame_list.sort()
	count_frame = 0
	for _frame_name in frame_list:
		frame_path = os.path.join(seq_path, _frame_name)
		save_path = os.path.join(out_dir, _frame_name)
		if not os.path.exists(save_path):
			img = cv2.imread(frame_path)
			if (img.shape[0] != 64) or (img.shape[1] != 64):
				img = cv2.resize(img, [64, 64], interpolation=cv2.INTER_CUBIC)

			if img is not None:
				# Save the cut img
				cv2.imwrite(save_path, img)
				count_frame += 1

	log_print(pid, FINISH,
			  'Contain %d valid frames. Saved to %s.'
			  % (count_frame, out_dir))


pool = Pool(WORKERS)
results = list()
pid = 0

print('Pretreatment Start.\n'
	  'Input path: %s\n'
	  'Output path: %s\n'
	  'Log file: %s\n'
	  'Worker num: %d' % (
		  INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))


# convert_videos('gray')
id_list = os.listdir(INPUT_PATH)
id_list.sort()
# Walk the input path
for _id in id_list:
	seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
	seq_type.sort()
	if MODE == 'test':
		seq_info = [_id]
		out_dir = os.path.join(OUTPUT_PATH, *seq_info)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		results.append(
			pool.apply_async(
				cut_pickle,
				args=(seq_info, pid)))
		sleep(0.02)
		pid += 1
	else:
		for _seq_type in seq_type:
			seq_info = [_id, _seq_type]
			out_dir = os.path.join(OUTPUT_PATH, *seq_info)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			results.append(
				pool.apply_async(
					cut_pickle,
					args=(seq_info, pid)))
			sleep(0.02)
			pid += 1

pool.close()
unfinish = 1
while unfinish > 0:
	unfinish = 0
	for i, res in enumerate(results):
		try:
			res.get(timeout=0.1)
		except Exception as e:
			if type(e) == MP_TimeoutError:
				unfinish += 1
				continue
			else:
				print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
					  i, type(e))
				raise e
pool.join()