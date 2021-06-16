# GLeNet_TF2.3

A complete inplementation of paper **《GLeNet-Global and Local Enhancement Networks for Paired and Unpaired Image Enhancement》**（2020 ECCV）

![image-20210611095200033](image_for_markdown/image-20210611095200033.png)

This repo is based on official code [GleNet](https://github.com/dongkwonjin/GleNet)，which is incomplete and brings trouble to me. So I rewrite the code for paird training.

官方代码比较简单，只有模型和数据，模型还不完整。我在官方代码的基础上，写了下训练的代码，目前只有 paired 训练，后面有时间再补充 Unpaired 训练的代码和模型。



目前只写了 GEN 全局增强网络部分；后面的 LEN 局部增强网络就一个 Unet 也没啥创新。其实，之前复现实验发现局部增强网络 LEN 就可以达到 25 的效果了，GEN + LEN 也就高个 0.3db 左右。估计就是全局网络求解一个近似值，然后 Unet 补一补。如果最后可以加一些平滑损失什么的，可能可以减少块效应。

损失函数 ，没啥好说的，MSE 和一个感知损失？感知损失没用。imagenet 训练的图片都相对单调，用在 fivek 这种相对复杂场景下不合适。只用了 MSE 做了个初步实验，结果过拟合了，训练 25.4db，测试 23.8db，不到 24。

近期事多，暂且搁置这个 repo，后面有时间再更新吧



## 环境

- Python3.7
- Tensorflow 2.3.0
- scikit-image 0.15.0
- ...



## Inference

```shell
python inference.py
```



## Data

使用的是 [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) 数据集的 **ExpertC** 图像对。本实验采用的 512px 分辨率（长宽最大 512，按比例下采样）的图像，下载链接如下：

1. [one drive](https://bupteducn-my.sharepoint.com/:u:/g/personal/fluence_dyf_bupt_edu_cn/EbbaJoJVSG9Guh5TWMLCXw8B0DkHPMwCGZ9QQeUtm6pwSA?e=FovmfI)

也可以使用其他数据集，低光照的数据集，比如 LOL 等。



## Train

```shell
python train.py
```

or

```shell
python weighted_train.py
```



## Validation

```shell
python test.py
```



## References

**papers**:

1. Kim, Hanul et al. “*Global and Local Enhancement Networks for Paired and Unpaired Image Enhancement*.” *ECCV* (2020).
2. Bychkovsky V, Paris S, Chan E, et al. *Learning photographic global tonal adjustment with a database of input/output image pairs*[C]//CVPR 2011. IEEE, 2011: 97-104.

**github**:

1. [GLeNet](https://github.com/dongkwonjin/GleNet)

**Dataset**

```txt
@inproceedings{fivek,
	author = "Vladimir Bychkovsky and Sylvain Paris and Eric Chan and Fr{\'e}do Durand",
	title = "Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs",
	booktitle = "The Twenty-Fourth IEEE Conference on Computer Vision and Pattern Recognition",
	year = "2011"
}
```

