"""
(c) MJMJ/2020
"""

import os
import copy

def find_latest_file_model(inputdir, pattern, epoch_max=1000):
	'''
	Searchs for check-points during training
	:param inputdir: path
	:param pattern: string compatible with format()
	:return: path to the best file, if any, "" otherwise
	'''

	if epoch_max < 0:
		maxepochs = 1000
	else:
		maxepochs = epoch_max

	bestfile = ""

	for epoch in range(1, maxepochs+1):
		modelname = os.path.join(inputdir, pattern.format(epoch))
		if os.path.isfile(modelname):
			bestfile = copy.deepcopy(modelname)


	return bestfile