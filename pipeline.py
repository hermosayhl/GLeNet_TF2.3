# Python
import os
import random
# 3rd party
import cv2
import numpy
# tensorflow
import tensorflow as tf




###########################################################################
#                                Data
###########################################################################





# 定义旋转函数
def cv2_rotate(image, angle=20, scale=0.9):
	height, width = image.shape[:2]   
	center = (width / 2, height / 2)  
	# 获得旋转矩阵
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	# 进行仿射变换，边界填充为255，即白色，默认为0，即黑色
	return cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))



def make_augment(low_quality, high_quality):
	# 以 0.6 的概率作数据增强
	if(random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if(cur_state == 'flip'):
				# 0.5 概率水平翻转
				if(random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif(cur_state == 'crop'):
				# 0.5 概率做裁剪
				if(random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.75, 0.95)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif(cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
	return low_quality, high_quality






def load_and_preprocess_augment(_image_path, _label_path, choice):

	low_quality = cv2.imread(str(_image_path.numpy())[2:-1])
	high_quality = cv2.imread(str(_label_path.numpy())[2:-1])

	# 作数据增强
	low_quality, high_quality = make_augment(low_quality, high_quality)

	# 是否要规范输入大小
	if(choice[0] == True):
		low_quality = cv2.resize(low_quality, (256, 256))
		high_quality = cv2.resize(high_quality, (256, 256))

	# numpy -> tensor
	low_quality = tf.convert_to_tensor(low_quality * 1. / 255, dtype=tf.float32)
	high_quality = tf.convert_to_tensor(high_quality * 1. / 255, dtype=tf.float32)
	
	return low_quality, high_quality




def get_dataloader(opt):

	image_list = os.listdir(os.path.join(opt.dataset_dir, opt.input_dir))
	assert image_list == os.listdir(os.path.join(opt.dataset_dir, opt.label_dir)), "images are not paired in {} and {}".format(opt.input_dir, opt.label_dir)
	
	random.shuffle(image_list)
	train_size = int(opt.dataset_ratios[0] * len(image_list))
	train_list = image_list[:train_size]
	valid_list = image_list[train_size:]

	train_image_list = [os.path.join(opt.dataset_dir, opt.input_dir, it) for it in train_list]
	train_label_list = [os.path.join(opt.dataset_dir, opt.label_dir, it) for it in train_list]
	valid_image_list = [os.path.join(opt.dataset_dir, opt.input_dir, it) for it in valid_list]
	valid_label_list = [os.path.join(opt.dataset_dir, opt.label_dir, it) for it in valid_list]

	print('train  :  {}\nvalid  :  {}'.format(len(train_list), len(valid_list)))
	
	# 数据集有点大的时候, 一开始存储路径, 每次用 map 函数读取和预处理图像; 注意 buffer_size 不能太大
	train_data = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
	train_data = train_data.map(lambda x, y: \
		tf.py_function(load_and_preprocess_augment, inp=[x, y, [opt.resize]], Tout=[tf.float32, tf.float32]))
	train_dataloader = train_data.shuffle(opt.buffer_size).batch(opt.train_batch_size)

	valid_data = tf.data.Dataset.from_tensor_slices((valid_image_list, valid_label_list))
	valid_data = valid_data.map(lambda x, y: \
		tf.py_function(load_and_preprocess_augment, inp=[x, y, [False]], Tout=[tf.float32, tf.float32]))
	valid_dataloader = valid_data.batch(opt.valid_batch_size).repeat(opt.valid_repeat)

	return train_dataloader, valid_dataloader, len(train_list), len(valid_list)