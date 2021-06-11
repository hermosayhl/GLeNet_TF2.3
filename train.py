# python
import os
import sys
import random
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 3rd party 
import numpy
import tensorflow as tf
print('tensorflow  :  {}'.format(tf.__version__))
# Self
import model
import evaluate
import pipeline

# 去除一些 future warning
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 设定 GPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print('GPU  :  {}'.format(gpus))
# 动态申请显存, 需要多少, 申请多少
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)










###########################################################################
#                                 config
###########################################################################

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_dir", type=str, default='/home/cgy/Chang/image_enhancement/datasets/fiveK', help="dirs of paired images for training")
	parser.add_argument("--input_dir", type=str, default='input', help="dirs of noise images")
	parser.add_argument("--label_dir", type=str, default='expertC_gt', help="dirs of retouched images")
	parser.add_argument("--buffer_size", type=int, default=128, help="buffer foor restrore images to forward")
	parser.add_argument("--new_size", type=tuple, default=(None, None), help="if resize, the size of image")
	parser.add_argument("--resize", type=bool, default=False, help="whether to resize images")
	parser.add_argument("--crop", type=bool, default=True, help="whether to crop images")
	parser.add_argument("--flip", type=bool, default=True, help="whether to flip images")
	parser.add_argument("--train_ratio", type=float, default=0.9, help="percentage of training images")
	parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training")
	parser.add_argument("--valid_batch_size", type=int, default=1, help="batch size for validation")
	parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for update gradient")
	parser.add_argument("--save_interval", type=int, default=1, help="interval for validation")
	parser.add_argument("--checkpoints_dir", type=str, default='./checkpoints/GleNet_simple_notResized', help="where to save trained models")
	parser.add_argument("--total_epochs", type=int, default=100, help="total nums of epoch")
	parser.add_argument("--with_testdataset", type=bool, default=False, help="whether to test")
	parser.add_argument("--network", type=str, default="GleNet", help="network architecture")
	parser.add_argument("--backbone", type=str, default="GEN", help="base network for 768 values")
	parser.add_argument("--residual", type=bool, default=True, help="whether to make residual connection")
	parser.add_argument("--save", type=bool, default=True, help="whether to save models")
	opt = parser.parse_args()
	if(opt.new_size == (None, None)): opt.resize = False
	if(opt.resize == True): opt.new_size = (256, 256)
	for l, r in vars(opt).items(): print('{}  :  {}'.format(l, r))
	return opt





	



if __name__ == '__main__':

	# 解析一些参数设置
	opt = get_args()

	# 获取数据读取器
	train_dataloader, valid_dataloader, train_len, valid_len = pipeline.get_dataloader(opt)

	# 定义网络结构
	network = model.GleNet(backbone=opt.backbone, residual=opt.residual)
	network.build(input_shape=(None, *opt.new_size, 3))

	# 参数变量
	parameters = network.trainable_variables

	# 优化器
	optimizer = tf.keras.optimizers.Adam(lr=opt.lr)

	# 保存的路径
	os.makedirs(opt.checkpoints_dir, exist_ok=True)

	# 开始训练
	max_psnr, max_ssim = 0.0, 0.0
	max_epoch = 0
	for epoch in range(1, opt.total_epochs + 1):
		# 训练一个 epoch
		mean_psnr, mean_ssim, mean_loss = 0, 0, 0
		# 读取数据
		for batch_num, (_image, _label) in enumerate(train_dataloader, 1):
			# 设置自动求导
			with tf.GradientTape() as tape:
				# 经过网络
				enhanced = network(_image)
				# 计算损失
				loss_value = tf.reduce_mean(tf.square(_label - enhanced))
				# 计算 PSNR
				psnr_value = evaluate.compute_psnr(enhanced, _label)
				ssim_value = evaluate.compute_ssim(enhanced, _label)
				# 统计一些值
				mean_loss += loss_value
				mean_psnr += psnr_value
				mean_ssim += ssim_value
			# 计算梯度
			grad = tape.gradient(loss_value, parameters)
			# 梯度更新
			optimizer.apply_gradients(zip(grad, parameters))
			# 输出一些信息
			sys.stdout.write('\r[Train===> epoch {}/{}] [batch {}/{}] [loss {:.5f}] [psnr {:.3f} - mean {:.3f}] [ssim {:.3f} - mean {:.3f}]'.format(
				epoch, opt.total_epochs, batch_num * opt.train_batch_size, train_len, \
				loss_value * (255 ** 2), \
				psnr_value, mean_psnr / batch_num, \
				ssim_value, mean_ssim / batch_num))
			
		train_psnr, train_ssim = mean_psnr / batch_num, mean_ssim / batch_num
		print()


		# 每隔多少个 epoch 验证一次
		if(epoch % opt.save_interval == 0):
			mean_psnr = 0
			mean_ssim = 0
			for batch_num, (_image, _label) in enumerate(valid_dataloader, 1):
				# 经过网络
				enhanced = network(_image)
				# 计算 PSNR
				psnr_value = evaluate.compute_psnr(enhanced, _label)
				ssim_value = evaluate.compute_ssim(enhanced, _label)
				# 统计一些值
				mean_psnr += psnr_value
				mean_ssim += ssim_value
				# 输出一些信息
				sys.stdout.write('\r[Valid===> epoch {}/{}] [batch {}/{}] [psnr {:.3f} - mean {:.3f}] [ssim {:.3f} - mean {:.3f}]'.format(
					epoch, opt.total_epochs, batch_num * opt.valid_batch_size, valid_len, \
					psnr_value, mean_psnr / batch_num, \
					ssim_value, mean_ssim / batch_num))
			# 记录当前最好的结果
			valid_psnr, valid_ssim = mean_psnr / batch_num, mean_ssim / batch_num
			if(valid_psnr > max_psnr): 
				max_psnr, max_epoch = valid_psnr, epoch
			if(valid_ssim > max_ssim): 
				max_ssim = valid_ssim
			# 保存模型
			if(opt.save == True):
				save_name = "GleNet_epoch_{}_train_{:.3f}_{:.3f}_valid_{:.3f}_{:.3f}_maxepoch_{}".format(
					epoch, train_psnr, train_ssim, valid_psnr, valid_ssim, max_epoch)
				os.makedirs(os.path.join(opt.checkpoints_dir, save_name), exist_ok=True)
				to_save = os.path.join(opt.checkpoints_dir, save_name, "GleNet")
				print('to_save  :  {}'.format(to_save))
				network.save_weights(to_save)
		print()

