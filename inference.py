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
opt.residual = True
opt.low_size = [256, 256]
# 训练参数
opt.use_cuda = True
opt.resize = False
# 实验参数
opt.images_dir = "./sample_imgs"
opt.results_dir = "./sample_results"
opt.checkpoints_file = "./checkpoints/batch_9/epoch_34_train_24.303_0.878_valid_23.972_0.857/GleNet"
for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.results_dir, exist_ok=True)
assert os.path.exists(opt.images_dir), "there are no images in folder {}".format(opt.images_dir)
assert os.path.exists(os.path.dirname(opt.checkpoints_file)), "checkpoints folder {} doesn't exist !".format(opt.checkpoints_file)


network = model.GleNet(backbone=opt.backbone, residual=opt.residual, low_size=opt.low_size)
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
		origin_tensor = pre_transform(origin)
		enhanced = network(origin_tensor, is_training=False)
		save_name = os.path.join(opt.results_dir, os.path.split(image_path)[-1])
		cv2.imwrite(save_name, post_transform(enhanced))
		print('{}/{}===>  processing {} and saved to {}'.format(cnt, len(images_list), image_path, save_name))
		



