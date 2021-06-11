# GLeNet_TF2.3

A complete inplementation of paper **《GLeNet-Global and Local Enhancement Networks for Paired and Unpaired Image Enhancement》**（2020 ECCV）

![image-20210611095200033](image_for_markdown/image-20210611095200033.png)

This repo is based on official code [GleNet](https://github.com/dongkwonjin/GleNet)，which is incomplete and brings trouble to me. So I rewrite the code for paird training.

官方代码比较简单，只有模型和数据，模型还不完整。我在官方代码的基础上，写了下训练的代码，目前只有 paired 训练，后面有时间再补充 Unpaired 训练的代码和模型。





## 环境







## Inference







## Data

使用的是 [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) 数据集的 **ExpertC** 图像对。本实验采用的 512px 分辨率（长宽最大 512，按比例下采样）的图像，下载链接如下：

1. [one drive](https://bupteducn-my.sharepoint.com/:u:/g/personal/fluence_dyf_bupt_edu_cn/EbbaJoJVSG9Guh5TWMLCXw8B0DkHPMwCGZ9QQeUtm6pwSA?e=FovmfI)

也可以使用其他数据集，低光照的数据集，比如 LOL 等。



## Train



## Validation





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

