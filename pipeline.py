# Python
import os
import random
# 3rd party
import cv2
# tensorflow
import tensorflow as tf




###########################################################################
#                                Data
###########################################################################



def read_image(path):
	image = tf.io.read_file(path)
	image = tf.io.decode_jpeg(image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype='float32')
	return image


def preprocess_image(x, y, choice):
	# 缩放到
	if(choice[0] == True):
		x = tf.image.resize(x, [256, 256])
		y = tf.image.resize(y, [256, 256])
	return x, y


def load_and_preprocess(_image_path, _label_path, choice):
	_image = read_image(_image_path)
	_label = read_image(_label_path)
	return preprocess_image(_image, _label, choice)




def load_and_preprocess_augment(_image_path, _label_path, choice):

	# 先读取图片
	x, y = load_and_preprocess(_image_path, _label_path, choice)

	# 一定概率横向翻转
	if(choice[1] == True and random.random() > 0.5):
		x = tf.image.flip_left_right(x)
		y = tf.image.flip_left_right(y)

	# 顺便搞个随机数 > 0.5, 随机概率进行裁剪
	if(choice[2] == True and random.random() > 0.3):
		H, W, _ = x.numpy().shape
		# 随机获取保持宽高比例的一个长度
		ratio = random.uniform(0.75, 0.95)
		_H = int(H * ratio)
		_W = int(W * ratio)
		# 生成一个坐标
		pos_lefttop = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
		x = tf.image.crop_to_bounding_box(x, pos_lefttop[0], pos_lefttop[1], _H, _W)
		y = tf.image.crop_to_bounding_box(y, pos_lefttop[0], pos_lefttop[1], _H, _W)
	return x, y



def show_image_tf2(__image, message):
	with tf.compat.v1.Session() as sess:
		image_rgb_float = sess.run(__image)
		plt.imshow(image_rgb_float)
		plt.title(message)
		plt.show()




def get_dataloader(opt):

	image_list = os.listdir(os.path.join(opt.dataset_dir, opt.input_dir))
	assert image_list == os.listdir(os.path.join(opt.dataset_dir, opt.label_dir)), "images are not paired in {} and {}".format(opt.input_dir, opt.label_dir)
	
	train_list = random.sample(image_list, int(opt.train_ratio * len(image_list)))
	valid_list = list(set(image_list) - set(train_list))
	train_image_list = [os.path.join(opt.dataset_dir, opt.input_dir, it) for it in train_list]
	train_label_list = [os.path.join(opt.dataset_dir, opt.label_dir, it) for it in train_list]
	valid_image_list = [os.path.join(opt.dataset_dir, opt.input_dir, it) for it in valid_list]
	valid_label_list = [os.path.join(opt.dataset_dir, opt.label_dir, it) for it in valid_list]
	
	# 如果还需要划分测试集
	if(opt.with_testdataset == True):
		test_image_list = train_image_list[:len(valid_list)]
		test_label_list = train_label_list[:len(valid_list)]
		train_image_list = train_image_list[len(valid_list):]
		train_label_list = train_label_list[len(valid_list):]

	print('train  :  {}\nvalid  :  {}'.format(len(train_list), len(valid_list)))
	if(opt.with_testdataset == True): 
		print('test  :  {}'.format(len(test_image_list)))

	# 数据集有点大的时候, 一开始存储路径, 每次用 map 函数读取和预处理图像; 注意 buffer_size 不能太大
	train_data = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
	train_data = train_data.map(lambda x, y: \
		tf.py_function(load_and_preprocess_augment, inp=[x, y, [opt.resize, opt.flip, False]], Tout=[tf.float32, tf.float32]))
	train_dataloader = train_data.shuffle(opt.buffer_size).batch(opt.train_batch_size)

	valid_data = tf.data.Dataset.from_tensor_slices((valid_image_list, valid_label_list))
	valid_data = valid_data.map(lambda x, y: \
		tf.py_function(load_and_preprocess, inp=[x, y, [opt.resize]], Tout=[tf.float32, tf.float32]))
	valid_dataloader = valid_data.batch(opt.valid_batch_size)
	# .cache(filename='./cache.tf-data') tensorflow 有缓存技术可以加速


	if(opt.with_testdataset == False):
		return train_dataloader, valid_dataloader, len(train_list), len(valid_list)
	else:
		test_data = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))
		test_data = test_data.map(lambda x, y: load_and_preprocess(x, y, opt))
		test_dataloader = test_data.batch(opt.valid_batch_size)
		return train_dataloader, valid_dataloader, test_dataloader, len(train_list), len(valid_list), len(test_image_list)

	