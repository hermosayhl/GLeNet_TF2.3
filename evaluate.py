# Python
# 3rd party
import cv2
import numpy
# tensorflow
import tensorflow as tf



###########################################################################
#                                Metrics
###########################################################################



class GleNetEvaluator():
	def __init__(self):
		self.mean_psnr = 0.0
		self.mean_ssim = 0.0
		self.mean_loss = 0.0
		self.mean_color_loss = 0.0
		self.mean_perceptual_loss = 0.0
		self.count = 0.0


	def update(self, label, enhanced):
		self.count += 1
		color_loss = tf.reduce_mean(tf.square(label - enhanced))
		self.mean_color_loss += color_loss
		psnr_value = tf.reduce_mean(tf.image.psnr(label, enhanced, max_val=1.0))
		self.mean_psnr += psnr_value
		ssim_value = tf.reduce_mean(tf.image.ssim(label, enhanced, max_val=1.0))
		self.mean_ssim += ssim_value
		# 损失加权
		total_loss = 1.0 * color_loss
		self.mean_loss += total_loss
		# 返回损失
		return total_loss


	def clear(self):
		self.count = 0
		self.mean_psnr, self.mean_ssim, self.mean_loss, self.mean_color_loss, self.mean_perceptual_loss \
			= 0.0, 0.0, 0.0, 0.0, .00


	def get(self):
		if(self.count == 0):
			return 0
		return self.mean_loss / self.count, \
				self.mean_color_loss * (255 ** 2) / self.count, \
				self.mean_perceptual_loss / self.count, \
				self.mean_psnr / self.count, \
				self.mean_ssim / self.count

