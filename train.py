# python
import os
import sys
import random
import argparse
# Self
import utils
utils.set_evironments(seed=212)
import model
import evaluate
import pipeline
# 3rd party 
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
opt.lr = 3e-4
opt.total_epochs = 100
opt.train_batch_size = 1
opt.valid_batch_size = 1
opt.valid_repeat = 4
opt.resize = False
opt.buffer_size = 32
# 实验参数
opt.exp_name = "simple"
opt.save = True
opt.valid_interval = 1
opt.checkpoints_dir = os.path.join("./checkpoints/", opt.exp_name)
opt.dataset_name = 'fivek'
opt.dataset_ratios = [0.9, 0.1]
opt.input_dir = "input"
opt.label_dir = "expertC_gt"
opt.dataset_dir = '/home/cgy/Chang/image_enhancement/datasets/fiveK'
# opt.dataset_dir = "C:/Code/HermosaWork/datasets/MIT-Adobe FiveK"
# 可视化参数
opt.visualize_size = 16
opt.visualize_batch = 100
opt.visualize_dir = os.path.join(opt.checkpoints_dir, 'train_phase') 


for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.visualize_dir, exist_ok=True)
assert os.path.exists(opt.dataset_dir), "dataset for low/high quality image pairs doesn't exist !"



	



if __name__ == '__main__':

	# 获取数据读取器
	train_dataloader, valid_dataloader, train_len, valid_len = pipeline.get_dataloader(opt)

	# 定义网络结构
	network = model.GleNet(backbone=opt.backbone, residual=opt.residual, low_size=opt.low_size)
	network.build(input_shape=(None, None, None, 3))

	# 参数变量
	parameters = network.trainable_variables

	# 优化器
	optimizer = tf.keras.optimizers.Adam(lr=opt.lr)

	# 保存的路径
	os.makedirs(opt.checkpoints_dir, exist_ok=True)

	# 评测, 计算损失
	train_evaluator = evaluate.GleNetEvaluator()


	# 开始训练
	for epoch in range(1, opt.total_epochs + 1):
		# 训练一个 epoch
		train_evaluator.clear()
		# 读取数据
		for batch_num, (_image, _label) in enumerate(train_dataloader, 1):
			# 设置自动求导
			with tf.GradientTape() as tape:
				# 经过网络
				enhanced = network(_image)
				# 计算损失
				loss_value = train_evaluator.update(_label, enhanced)
				# 这里可以加一个感知损失, 就用我那个卷积分类的特征, 可视化下特征
			# 计算梯度
			grad = tape.gradient(loss_value, parameters)
			# 梯度更新
			optimizer.apply_gradients(zip(grad, parameters))
			# 输出一些信息
			sys.stdout.write('\r[Train===> epoch {}/{}] [batch {}/{}] [loss {:.4f}] [color {:.3f}] [perceptual {:.3f}] [psnr {:.3f}] [ssim {:.3f}]'.format(
				epoch, opt.total_epochs, batch_num * opt.train_batch_size, train_len, \
				*train_evaluator.get()))

		train_loss, train_color_loss, train_perceptual_loss, train_psnr, train_ssim = train_evaluator.get()
		print()


		# 每隔多少个 epoch 验证一次
		if(epoch % opt.valid_interval == 0):
			valid_evaluator = evaluate.GleNetEvaluator()
			for batch_num, (_image, _label) in enumerate(valid_dataloader, 1):
				# 经过网络
				enhanced = network(_image)
				# 计算损失
				loss_value = valid_evaluator.update(_label, enhanced)
				# 输出一些信息
				sys.stdout.write('\r[Valid===> epoch {}/{}] [batch {}/{}] [loss {:.4f}] [color {:.3f}] [perceptual {:.3f}] [psnr {:.3f}] [ssim {:.3f}]'.format(
					epoch, opt.total_epochs, batch_num * opt.valid_batch_size, valid_len, \
					*valid_evaluator.get()))
			# 记录当前最好的结果
			valid_loss, valid_color_loss, valid_perceptual_loss, valid_psnr, valid_ssim = valid_evaluator.get()
			
			# 保存模型
			if(opt.save == True):
				save_name = "epoch_{}_train_{:.3f}_{:.3f}_valid_{:.3f}_{:.3f}".format(
					epoch, train_psnr, train_ssim, valid_psnr, valid_ssim)
				os.makedirs(os.path.join(opt.checkpoints_dir, save_name), exist_ok=True)
				to_save = os.path.join(opt.checkpoints_dir, save_name, "GleNet")
				print('to_save  :  {}'.format(to_save))
				network.save_weights(to_save)
		print()

