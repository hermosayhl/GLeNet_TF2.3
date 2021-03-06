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
import skimage
import dill as pickle
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
opt.dataset_dir = "/home/cgy/Chang/image_enhancement/datasets/fiveK"
if(opt.use_local == False):
	opt.config_file = "./checkpoints/LEN_False_batch_4/options.pkl"
	opt.checkpoints_file = './checkpoints/LEN_False_batch_4/epoch_71_train_24.595_0.885_valid_23.945_0.862/GleNet'
else:
	opt.config_file = "./checkpoints/LEN_True_batch_4/options.pkl"
	opt.checkpoints_file = "./checkpoints/LEN_True_batch_4/epoch_88_train_25.148_0.904_valid_24.535_0.887/GleNet"
for l, r in vars(opt).items(): print(l, " : ", r)
assert os.path.exists(os.path.dirname(opt.checkpoints_file)), "checkpoints folder {} doesn't exists !".format(opt.checkpoints_file)
assert os.path.exists(opt.dataset_dir), "dataset folder {} doesn't exists !".format(opt.dataset_dir)




network = model.GleNet(backbone=opt.backbone, low_size=opt.low_size, \
	use_local=opt.use_local)
network.built = True
network.load_weights(opt.checkpoints_file)
print('loaded weights from {}'.format(opt.checkpoints_file))




with open(opt.config_file, 'rb') as reader:
	config = pickle.load(reader)

images_list = config['split_data']['valid_image_list']
labels_list = config['split_data']['valid_label_list']
print('{} images are to be processed !'.format(len(images_list)))



pre_transform = lambda x: tf.expand_dims(tf.convert_to_tensor(x * 1. / 255, dtype=tf.float32), axis=0) 
post_transform = lambda x: (tf.squeeze(x).numpy() * 255).astype('uint8')

mean_psnr = 0
mean_ssim = 0
with utils.Timer() as time_scope:
	for cnt, image_path in enumerate(images_list, 1):
		origin = cv2.imread(image_path)
		label = cv2.imread(labels_list[cnt - 1])
		if(opt.use_local == True):
			origin = pipeline.crop_as_8(origin)
			label = pipeline.crop_as_8(label)
		origin_tensor = pre_transform(origin)
		enhanced = network(origin_tensor, is_training=False)
		enhanced = post_transform(enhanced)
		psnr_value = skimage.measure.compare_psnr(enhanced, label)
		ssim_value = skimage.measure.compare_ssim(enhanced, label, multichannel=True)
		mean_psnr += psnr_value
		mean_ssim += ssim_value
		print('{}/{}===> [psnr {:.3f}] [ssim {:.3f}]  processing {}'.format(
			cnt, len(images_list), mean_psnr / cnt, mean_ssim / cnt, image_path))
		



