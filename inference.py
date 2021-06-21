# python
import os
import sys
import random
import argparse
# Self
import utils
utils.set_evironments(seed=212)
import model
import pipeline
# 3rd party 
import cv2
import numpy
import tensorflow as tf



# 参数
opt = lambda: None
# 网络参数
opt.backbone = "GEN"
opt.low_size = [256, 256]
# 训练参数
opt.use_cuda = True
opt.resize = False
# 实验参数
opt.use_local = False
opt.images_dir = "./sample_imgs"
# opt.images_dir = "./display/input"

if(opt.use_local == False):
	opt.results_dir = "./sample_results_GEN"
	opt.checkpoints_file = './checkpoints/LEN_False_batch_4/epoch_71_train_24.595_0.885_valid_23.945_0.862/GleNet'
else:
	opt.results_dir = "./sample_results"
	opt.checkpoints_file = "./checkpoints/LEN_True_batch_4/epoch_88_train_25.148_0.904_valid_24.535_0.887/GleNet"

for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.results_dir, exist_ok=True)
assert os.path.exists(opt.images_dir), "there are no images in folder {}".format(opt.images_dir)
assert os.path.exists(os.path.dirname(opt.checkpoints_file)), "checkpoints folder {} doesn't exist !".format(opt.checkpoints_file)


network = model.GleNet(backbone=opt.backbone, low_size=opt.low_size, \
	use_local=opt.use_local)
network.built = True
network.load_weights(opt.checkpoints_file)
print('loaded weights from {}'.format(opt.checkpoints_file))


images_list = os.listdir(opt.images_dir)
images_list = [it for it in images_list if(it.lower().endswith(
	('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')))]
images_list = [os.path.join(opt.images_dir, it) for it in images_list]
print('{} images are to be processed !'.format(len(images_list)))


pre_transform = lambda x: tf.expand_dims(tf.convert_to_tensor(x * 1. / 255, dtype=tf.float32), axis=0) 
post_transform = lambda x: (tf.squeeze(x).numpy() * 255).astype('uint8')


with utils.Timer() as time_scope:
	for cnt, image_path in enumerate(images_list, 1):
		origin = cv2.imread(image_path)
		# 如果使用了局部增强网络, 就要把图片裁剪成 8 的倍数
		if(opt.use_local == True):
			origin = pipeline.crop_as_8(origin)
		origin_tensor = pre_transform(origin)
		enhanced = network(origin_tensor, is_training=False)
		save_name = os.path.join(opt.results_dir, os.path.split(image_path)[-1])
		cv2.imwrite(save_name, post_transform(enhanced))
		print('{}/{}===>  processing {} and saved to {}'.format(cnt, len(images_list), image_path, save_name))
		



