# SRMD Pytorch

[![last-commit](https://img.shields.io/github/last-commit/HollowMan6/SRMD-Pytorch)](../../graphs/commit-activity)
![Python package](https://github.com/HollowMan6/SRMD-Pytorch/workflows/Python%20package/badge.svg)

[![Followers](https://img.shields.io/github/followers/HollowMan6?style=social)](https://github.com/HollowMan6?tab=followers)
[![watchers](https://img.shields.io/github/watchers/HollowMan6/SRMD-Pytorch?style=social)](../../watchers)
[![stars](https://img.shields.io/github/stars/HollowMan6/SRMD-Pytorch?style=social)](../../stargazers)
[![forks](https://img.shields.io/github/forks/HollowMan6/SRMD-Pytorch?style=social)](../../network/members)

[![Open Source Love](https://img.shields.io/badge/-%E2%9D%A4%20Open%20Source-Green?style=flat-square&logo=Github&logoColor=white&link=https://hollowman6.github.io/fund.html)](https://hollowman6.github.io/fund.html)
[![MIT Licence](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/mit-license.php)
[![Repo-Size](https://img.shields.io/github/repo-size/HollowMan6/SRMD-Pytorch.svg)](../../archive/master.zip)

[![Total alerts](https://img.shields.io/lgtm/alerts/g/HollowMan6/SRMD-Pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HollowMan6/SRMD-Pytorch/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HollowMan6/SRMD-Pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HollowMan6/SRMD-Pytorch/context:python)

(English version is down below)

[Python库依赖](../../network/dependencies)(建议在[Anaconda](https://www.anaconda.com)环境下操作)

本仓库改编于原作者项目[cszn/KAIR](https://github.com/cszn/KAIR)

相关论文：CVPR 2018 [Learning a Single Convolutional Super-Resolution Network for Multiple Degradations](https://www.researchgate.net/profile/Wangmeng_Zuo/publication/321902177_Learning_a_Single_Convolutional_Super-Resolution_Network_for_Multiple_Degradations/links/5a3a5dfe458515889d2dd92e/Learning-a-Single-Convolutional-Super-Resolution-Network-for-Multiple-Degradations.pdf)

模型结构：

![](https://github.com/cszn/SRMD/raw/master/figs/architecture.png)

SRMD已经训练的模型保存在[`model_zoo`](model_zoo)中。(模型来源于原作者 https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D)

***SRMD模型***：
* srmd_x2.pth
* srmd_x3.pth
* srmd_x4.pth

**输入**：19维的数据，其中15维为经过PCA降维后进行维度拉伸的模糊核，还有1个维度为图片噪声维度，另外3个维度分别为图片的RGB通道，即模型输入为：(19,图片宽,图片高)

图片经过GAN网络的处理，最后进行PixelShuffle，放大指定的倍数。

**输出**：放大了的图片的RGB通道，即模型输出为：(3,图片宽,图片高)


***SRMD Noise Free模型***：
* srmdnf_x2.pth
* srmdnf_x3.pth
* srmdnf_x4.pth

**输入**：18维的数据，其中15维为经过PCA降维后进行维度拉伸的模糊核，另外3个维度分别为图片的RGB通道，即模型输入为：(18,图片宽,图片高)

图片经过GAN网络的处理，最后进行PixelShuffle，放大指定的倍数。

**输出**：图片的RGB通道，即模型输出为：(3,图片宽,图片高)

## 训练

[`options/train_srmd.json`](options/train_srmd.json)中定义了模型的参数，可以进行修改调参。同时其中也给定了训练集和测试集路径的定义等。

按照默认配置，请向[`trainsets/trainH`](trainsets/trainH)中添加高质量图片来进行模型的训练。

[`testsets/set5`](testsets/set5)为测试集。

直接运行[`main_train_srmd.py`](main_train_srmd.py)即可进行训练。

训练日志保存在[`superresolution/srmd/train.log`](superresolution/srmd/train.log)，每次训练开始时的配置保存在[`superresolution/srmd/options`](superresolution/srmd/options)文件夹中。

[`options/train_srmd.json`](options/train_srmd.json)中`checkpoint_test`配置每进行5个epoch训练后进行测试，并将测试生成的图片保存在[`superresolution/srmd/images`](superresolution/srmd/images)中;`checkpoint_save`配置每进行5个epoch训练后保存当前模型。

训练产生的模型保存在[`superresolution/srmd/models`](superresolution/srmd/models)文件夹中。

## 测试

直接运行[`main_test_srmd.py`](main_test_srmd.py)即可对指定模型进行测试。

按照默认配置，默认[`testsets/set5`](testsets/set5)为测试集。

请确保PCA降维数据文件[`kernels/srmd_pca_matlab.mat`](kernels/srmd_pca_matlab.mat)存在（使用该文件预定义的参数对模糊核进行PCA降维，准备处理输入数据）。

测试结果和记录存放在[`results`](results)中。

## 查看网络描述

直接运行[`describe_model.py`](describe_model.py)即可查看每个SRMD模型(6个)的网络结构和输出。

## 转换为ONNX/NCNN模型

确保SRMD已经训练的模型保存在[`model_zoo`](model_zoo)中，直接运行[`pytorch2onnx.py`](pytorch2onnx.py)即可将pytorch模型转为onnx模型。

转换后的模型存放在[`onnx_models`](onnx_models)中。

随后步骤参考：https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md

使用`onnx-simplifier`简化生成后的ONNX模型：

```bash
cd onnx_model
pip install onnx-simplifier
python -m onnxsim srmd_x2.onnx srmd_x2-sim.onnx
python -m onnxsim srmd_x3.onnx srmd_x3-sim.onnx
python -m onnxsim srmd_x4.onnx srmd_x4-sim.onnx
python -m onnxsim srmdnf_x2.onnx srmdnf_x2-sim.onnx
python -m onnxsim srmdnf_x3.onnx srmdnf_x3-sim.onnx
python -m onnxsim srmdnf_x4.onnx srmdnf_x4-sim.onnx
```

然后就可以使用编译好的NCNN工具将ONNX模型转化成NCNN模型：

```bash
onnx2ncnn srmd_x2-sim.onnx srmd_x2.param srmd_x2.bin
onnx2ncnn srmd_x3-sim.onnx srmd_x3.param srmd_x2.bin
onnx2ncnn srmd_x4-sim.onnx srmd_x4.param srmd_x2.bin
onnx2ncnn srmdnf_x2-sim.onnx srmdnf_x2.param srmd_x2.bin
onnx2ncnn srmdnf_x3-sim.onnx srmdnf_x3.param srmd_x2.bin
onnx2ncnn srmdnf_x4-sim.onnx srmdnf_x4.param srmd_x2.bin
```

# SRMD Pytorch

[Python Dependencies](../../network/dependencies) (Recommend running under [Anaconda](https://www.anaconda.com))

This repository is derived from the original author project [cszn/KAIR](https://github.com/cszn/KAIR).

Related papers: CVPR 2018 [Learning a Single Convolutional Super-Resolution Network for Multiple Degradations](https://www.researchgate.net/profile/Wangmeng_Zuo/publication/321902177_Learning_a_Single_Convolutional_Super-Resolution_Network_for_Multiple_Degradations/links/5a3a5dfe458515889d2dd92e/Learning-a-Single-Convolutional-Super-Resolution-Network-for-Multiple-Degradations.pdf)

Model structure:

![](https://github.com/cszn/SRMD/raw/master/figs/architecture.png)

SRMD trained models are saved in [`model_zoo`](model_zoo). (The model comes from the original author https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D )

***SRMD model***：

* srmd_x2.pth
* srmd_x3.pth
* srmd_x4.pth

**Input**: 19 dimensional data, of which 15 dimensions are blur kernels after dimension reduction by PCA, 1 dimension is image noise dimension, and the other 3 dimensions are RGB channels of images, so the model input is: (19, image width, image height)

The image is processed by GAN network, and finally PixelSuffle is performed to enlarge the specified pictures.

**Output**: the RGB channel of the enlarged image, so the model output is: (3, picture width, image height)

***SRMD Noise Free model***：

* srmdnf_x2.pth
* srmdnf_x3.pth
* srmdnf_x4.pth

**Input**: 18 dimensional data, of which 15 dimensions are blur kernels after dimension reduction by PCA, and the other 3 dimensions are RGB channels of images, so the model input is: (18, image width, image height)

The image is processed by GAN network, and finally PixelSuffle is performed to enlarge the specified pictures.

**Output**: the RGB channel of the enlarged image, so the model output is: (3, picture width, image height)

## Training

Parameters of the model are defined in [`options/train_srmd.json`](options/train_srmd.json), which can be modified and adjusted. At the same time, the definitions of training set and test set path etc. are given.

According to the default configuration, please add high-quality pictures to [`trainsets/trainH`](trainsets/trainH) to train the model.

[`testsets/set5`](testsets/set5) is the test set.

Run [`main_train_srmd.py`](main_train_srmd.py) directly and then you can train it.

The training log is saved in [`superresolution/srmd/train.log`](superresolution/srmd/train.log). At the beginning of each training, the configuration is saved in the [`superresolution/srmd/options`](superresolution/srmd/options) folder.

In [`options/train_srmd.json`](options/train_srmd.json), `checkpoint_Test` configures to test after every 5 epoch training, and save the images generated by the test in [`superresolution/srmd/images`](superresolution/srmd/images), `checkpoint_save` configures to save the current model after every 5 epoch training.

The model generated by training is saved in the folder [`superresolution/srmd/models`](superresolution/srmd/models).

## Testing

Run [`main_test_srmd.py`](main_test_srmd.py) directly, and then you can test the specified model.

According to the default configuration, [`testsets/set5`](testsets/set5) is the test set by default.

Please ensure that the PCA dimension reduction data stores in [`kernels/srmd_pca_matlab.mat`](kernels/srmd_pca_matlab.mat).(It uses the predefined parameters of the file to reduce the dimension of blur kernel by PCA, and prepare to process the input data).

Test results and records are stored in [`results`](results).

## View network description

Run [`describe_model.py`](describe_model.py) directly, and then you can view the network structure and output of each SRMD model (6 models).

## Convert to ONNX / NCNN model

Make sure that the SRMD trained model is saved in the [`model_zoo`](model_zoo), run [`pytorch2onnx.py`](pytorch2onnx.py) directly, and then model can be transformed into ONNX model.

The converted model is stored in [`onnx_models`](onnx_models).

the following steps refer to: https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md

Use `onnx simplifier` to simplify the generated onnx model:

```bash
cd onnx_model
pip install onnx-simplifier
python -m onnxsim srmd_x2.onnx srmd_x2-sim.onnx
python -m onnxsim srmd_x3.onnx srmd_x3-sim.onnx
python -m onnxsim srmd_x4.onnx srmd_x4-sim.onnx
python -m onnxsim srmdnf_x2.onnx srmdnf_x2-sim.onnx
python -m onnxsim srmdnf_x3.onnx srmdnf_x3-sim.onnx
python -m onnxsim srmdnf_x4.onnx srmdnf_x4-sim.onnx
```

Then, the compiled NCNN tools can be used to convert the ONNX model into the NCNN model:

```bash
onnx2ncnn srmd_x2-sim.onnx srmd_x2.param srmd_x2.bin
onnx2ncnn srmd_x3-sim.onnx srmd_x3.param srmd_x2.bin
onnx2ncnn srmd_x4-sim.onnx srmd_x4.param srmd_x2.bin
onnx2ncnn srmdnf_x2-sim.onnx srmdnf_x2.param srmd_x2.bin
onnx2ncnn srmdnf_x3-sim.onnx srmdnf_x3.param srmd_x2.bin
onnx2ncnn srmdnf_x4-sim.onnx srmdnf_x4.param srmd_x2.bin
```
