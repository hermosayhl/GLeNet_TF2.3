# Python
import os
import math
import random
import datetime
# 3rrd party
import cv2
import numpy





def set_evironments(seed=212):
	
	# 设定随机数、GPU、log 级别
	random.seed(seed)
	numpy.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	# Tensorflow
	import tensorflow as tf
	print('tensorflow  :  {}'.format(tf.__version__))
	# 去除一些 future warning
	old_v = tf.compat.v1.logging.get_verbosity()
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	# 设定 GPU
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	print('GPU  :  {}'.format(gpus))
	# 动态申请显存, 需要多少, 申请多少
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)







class Timer:
	def __enter__(self):
		self.start = datetime.datetime.now()

	def __exit__(self, type, value, trace):
		_end = datetime.datetime.now()
		print('耗时  :  {}'.format(_end - self.start))


