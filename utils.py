# Python
import os
import math
import random
import datetime
# 3rrd party
import cv2
import numpy





def set_evironments(seed=212):

	# GPU、log 级别
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	
	# 设定随机数, 虽然并没有什么用, tensorflow 2.3 不支持
	random.seed(seed)
	numpy.random.seed(seed)
	
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC']='1'
	os.environ['TF_KERAS'] = '1'
	os.environ['PYTHONHASHSEED'] = str(seed)
	import tensorflow as tf
	tf.random.set_seed(seed)

	# tensorflow
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


