# SRMD Pytorch

[![last-commit](https://img.shields.io/github/last-commit/HollowMan6/SRMD-Pytorch)](../../graphs/commit-activity)
![Python package](https://github.com/HollowMan6/SRMD-Pytorch/workflows/Python%20package/badge.svg)

[![Followers](https://img.shields.io/github/followers/HollowMan6?style=social)](https://github.com/HollowMan6?tab=followers)
[![watchers](https://img.shields.io/github/watchers/HollowMan6/SRMD-Pytorch?style=social)](../../watchers)
[![stars](https://img.shields.io/github/stars/HollowMan6/SRMD-Pytorch?style=social)](../../stargazers)
[![forks](https://img.shields.io/github/forks/HollowMan6/SRMD-Pytorch?style=social)](../../network/members)

[![Open Source Love](https://img.shields.io/badge/-%E2%9D%A4%20Open%20Source-Green?style=flat-square&logo=Github&logoColor=white&link=https://hollowman6.github.io/fund.html)](https://hollowman6.github.io/fund.html)
[![GPL Licence](https://img.shields.io/badge/license-GPL-blue)](https://opensource.org/licenses/GPL-3.0/)
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

## 使用模型

***本脚本使用原作者训练的模型进行预测，如有需要请自行训练模型，自行定义模糊核和PCA降维数据。***

**使用双三次插值方法实现Alpha（透明）通道的放大，弥补了源模型不支持透明通道的缺憾。**

使用方法：

```text
main_srmd.py -i 输入路径 -o 输出路径 [选项]...

  -h help              显示这个帮助
  -i input-path        输入图片路径 (jpg/png/...) 或文件夹 (默认=运行文件夹)
  -o output-path       输出图片路径 (jpg/png/...) 或文件夹 (默认=运行文件夹)
  -n noise-level       降噪等级 (-1/0/1/2/3/4/5/6/7/8/9/10, 默认=3)
  -s scale             放大比例 (2/3/4, 默认=2)
  -m model-path        srmd 模型路径 (默认='model_zoo')
  -p srmd-pca-path     srmd 模糊核 pca 数据路径 (默认='kernels/srmd_pca_matlab.mat')
  -x tta               开启 x8 性能模式 (默认关闭)
  -c cuda              开启 CUDA GPU 计算 (默认关闭)
  -f format            输出图片格式 (jpg/png/..., 默认=png)

```

***支持的图片格式 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif'***

***注：X8性能模式（又称TTA模式）为讲图片另外进行7种不同的旋转等数据增强操作，然后取放大后各个像素点平均值，从而使得图片质量更佳，但是会比原来慢8倍***

- `input-path`和`output-path`接受文件路径或目录路径
- `noise-level`=降噪等级，值越大表示去噪效果越强，-1=不降噪
- `scale`=放大比例，2=放大2x，3=放大3x，4=放大4x

直接运行[`main_srmd.py`](main_srmd.py)即可进行模型的使用来对图片进行放大。

### 运行过程中出现 RuntimeError: CUDA error: unspecified launch failure

此问题在开启`-c`参数 CUDA GPU 计算时在处理分辨率较大的图片会出现此问题，一般为显存溢出导致，建议通过运行时不加`-c`关闭 CUDA GPU 计算。

## 训练模型

[`options/train_srmd.json`](options/train_srmd.json)中定义了模型的参数，可以进行修改调参。同时其中也给定了训练集和测试集路径的定义等。

按照默认配置，请向[`trainsets/trainH`](trainsets/trainH)中添加高质量图片来进行模型的训练。

[`testsets/set5`](testsets/set5)为测试集。

直接运行[`main_train_srmd.py`](main_train_srmd.py)即可进行训练。

训练日志保存在[`superresolution/srmd/train.log`](superresolution/srmd/train.log)，每次训练开始时的配置保存在[`superresolution/srmd/options`](superresolution/srmd/options)文件夹中。

[`options/train_srmd.json`](options/train_srmd.json)中`checkpoint_test`配置每进行5个epoch训练后进行测试，并将测试生成的图片保存在[`superresolution/srmd/images`](superresolution/srmd/images)中;`checkpoint_save`配置每进行5个epoch训练后保存当前模型。

训练产生的模型保存在[`superresolution/srmd/models`](superresolution/srmd/models)文件夹中。

***温馨提示：如果设置了`batch_size>1`，请保证训练集中图片分辨率相同，且个数超过一个batch_size大小，否则建议`batch_size`值设为1。***

## 测试模型

直接运行[`main_test_srmd.py`](main_test_srmd.py)即可对指定模型进行测试。

按照默认配置，默认[`testsets/set5`](testsets/set5)为测试集。

请确保PCA降维数据文件[`kernels/srmd_pca_matlab.mat`](kernels/srmd_pca_matlab.mat)存在（使用该文件预定义的参数对模糊核进行PCA降维，准备处理输入数据）。

测试结果和记录存放在[`results`](results)中。

## 查看模型描述

直接运行[`describe_model.py`](describe_model.py)即可查看每个SRMD模型(6个)的网络结构和输出。

## 转换为ONNX/NCNN模型

确保SRMD已经训练的模型保存在[`model_zoo`](model_zoo)中，直接运行[`pytorch2onnx.py`](pytorch2onnx.py)即可将pytorch模型转为onnx模型。

转换后的模型存放在[`onnx_models`](onnx_models)中。

随后步骤参考：https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md

使用`onnx-simplifier`简化生成后的ONNX模型：

```bash
cd onnx_models
pip install onnx-simplifier
python -m onnxsim srmd_x2.onnx srmd_x2-sim.onnx
python -m onnxsim srmd_x3.onnx srmd_x3-sim.onnx
python -m onnxsim srmd_x4.onnx srmd_x4-sim.onnx
python -m onnxsim srmdnf_x2.onnx srmdnf_x2-sim.onnx
python -m onnxsim srmdnf_x3.onnx srmdnf_x3-sim.onnx
python -m onnxsim srmdnf_x4.onnx srmdnf_x4-sim.onnx
```

然后就可以使用编译好的NCNN工具将ONNX模型转化成NCNN模型。

这里我在[onnx2ncnn](onnx2ncnn)文件夹下准备了在Win64环境下通过使用[windows-vs2019-avx2 CI](https://github.com/Tencent/ncnn/blob/36a591da8365f3fbb33f995c4303f49d47c4d553/.github/workflows/windows-x64-cpu-vs2019.yml#L45-L56) `MSVC 19.27.29112.0`环境编译[NCNN #36a591d](https://github.com/Tencent/ncnn/tree/36a591da8365f3fbb33f995c4303f49d47c4d553)仓库得到的NCNN工具可执行文件[onnx2ncnn.exe](onnx2ncnn/onnx2ncnn.exe)。如果你的系统为Win64，在确保上述步骤已经完成的情况下，可以直接双击执行脚本[convert.cmd](onnx2ncnn/convert.cmd)，会直接在`onnx2ncnn/srmd_ncnn_models`文件夹下自动生成SRMD NCNN模型文件。

否则请自行编译NCNN，得到onnx2ncnn工具。

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

## Use Model

***This script uses the model trained by the original author for prediction. If necessary, please train the model, define blur kernel and PCA dimension reduction data by yourself.***

**The bicubic interpolation method is used to enlarge the alpha channel, which makes up for the defect that the source model does not support transparent (Alpha) channel.**

Usage：

```text
main_srmd.py -i infile -o outfile [options]...

  -h help              show this help
  -i input-path        input image path (jpg/png/...) or directory (default=running directory)
  -o output-path       output image path (jpg/png/...) or directory (default=running directory)
  -n noise-level       denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)
  -s scale             upscale ratio (2/3/4, default=2)
  -m model-path        srmd model path (default='model_zoo')
  -p srmd-pca-path     srmd blur kernel pca data path (default='kernels/srmd_pca_matlab.mat')
  -x tta               enable x8 performance mode (default disabled)
  -c cuda              enable CUDA GPU caculating (default disabled)
  -f format            output image format (jpg/png/..., default=png)

```

***Supported image extensions 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif'***

***Note: the X8 performance mode (also known as TTA mode) is to perform seven different kinds of data augment operations such as rotation etc., and then take the average value of each pixel after upscale. So as to improve the image quality, it will be 8 times slower than the original mode.***

- `input-path` and `output-path` accept either file path or directory path
- `noise-level` = noise level, larger value means stronger denoise effect, -1 = no effect
- `scale` = scale level, 2 = upscale 2x, 3 = upscale 3x, 4 = upscale 4x

Run [`main_srmd.py`](main_srmd.py)irectly and then you can use the model to enlarge the picture。

### Encounter `RuntimeError: CUDA error: unspecified launch failure`

This problem occurs when the '-c' parameter CUDA GPU caculating is turned on. It is usually caused by GPU memory overflow. It is recommended to turn off CUDA GPU calculation by without adding '-c' during runtime when this occurs.

## Training Models

Parameters of the model are defined in [`options/train_srmd.json`](options/train_srmd.json), which can be modified and adjusted. At the same time, the definitions of training set and test set path etc. are given.

According to the default configuration, please add high-quality pictures to [`trainsets/trainH`](trainsets/trainH) to train the model.

[`testsets/set5`](testsets/set5) is the test set.

Run [`main_train_srmd.py`](main_train_srmd.py) directly and then you can train it.

The training log is saved in [`superresolution/srmd/train.log`](superresolution/srmd/train.log). At the beginning of each training, the configuration is saved in the [`superresolution/srmd/options`](superresolution/srmd/options) folder.

In [`options/train_srmd.json`](options/train_srmd.json), `checkpoint_Test` configures to test after every 5 epoch training, and save the images generated by the test in [`superresolution/srmd/images`](superresolution/srmd/images), `checkpoint_save` configures to save the current model after every 5 epoch training.

The model generated by training is saved in the folder [`superresolution/srmd/models`](superresolution/srmd/models).

***Warm tip: if you set the `batch_size > 1`. Please make sure that the resolution of images in the training set is the same, and the number of pictures exceeds one batch_size, otherwise it is suggested that `batch_size` value is set to 1.***

## Testing Models

Run [`main_test_srmd.py`](main_test_srmd.py) directly, and then you can test the specified model.

According to the default configuration, [`testsets/set5`](testsets/set5) is the test set by default.

Please ensure that the PCA dimension reduction data stores in [`kernels/srmd_pca_matlab.mat`](kernels/srmd_pca_matlab.mat).(It uses the predefined parameters of the file to reduce the dimension of blur kernel by PCA, and prepare to process the input data).

Test results and records are stored in [`results`](results).

## View Models description

Run [`describe_model.py`](describe_model.py) directly, and then you can view the network structure and output of each SRMD model (6 models).

## Convert to ONNX / NCNN model

Make sure that the SRMD trained model is saved in the [`model_zoo`](model_zoo), run [`pytorch2onnx.py`](pytorch2onnx.py) directly, and then model can be transformed into ONNX model.

The converted model is stored in [`onnx_models`](onnx_models).

the following steps refer to: https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md

Use `onnx simplifier` to simplify the generated onnx model:

```bash
cd onnx_models
pip install onnx-simplifier
python -m onnxsim srmd_x2.onnx srmd_x2-sim.onnx
python -m onnxsim srmd_x3.onnx srmd_x3-sim.onnx
python -m onnxsim srmd_x4.onnx srmd_x4-sim.onnx
python -m onnxsim srmdnf_x2.onnx srmdnf_x2-sim.onnx
python -m onnxsim srmdnf_x3.onnx srmdnf_x3-sim.onnx
python -m onnxsim srmdnf_x4.onnx srmdnf_x4-sim.onnx
```

Then, the compiled NCNN tools can be used to convert the ONNX model into the NCNN model.

Here I have offered [onnx2ncnn.exe](onnx2ncnn/onnx2ncnn.exe) binary file in [onnx2ncnn](onnx2ncnn) folder which is for Win64 Compiled using [windows-vs2019-avx2 CI](https://github.com/Tencent/ncnn/blob/36a591da8365f3fbb33f995c4303f49d47c4d553/.github/workflows/windows-x64-cpu-vs2019.yml#L45-L56) and [NCNN #36a591d](https://github.com/Tencent/ncnn/tree/36a591da8365f3fbb33f995c4303f49d47c4d553) with `MSVC 19.27.29112.0`. If your environment is Win64, and have completed the steps above, you can directly run [convert.cmd](onnx2ncnn/convert.cmd). Then the SRMD models will be generated under `onnx2ncnn/srmd_ncnn_models`.

Otherwise please compile NCNN for your own to get onnx2ncnn.

```bash
onnx2ncnn srmd_x2-sim.onnx srmd_x2.param srmd_x2.bin
onnx2ncnn srmd_x3-sim.onnx srmd_x3.param srmd_x2.bin
onnx2ncnn srmd_x4-sim.onnx srmd_x4.param srmd_x2.bin
onnx2ncnn srmdnf_x2-sim.onnx srmdnf_x2.param srmd_x2.bin
onnx2ncnn srmdnf_x3-sim.onnx srmdnf_x3.param srmd_x2.bin
onnx2ncnn srmdnf_x4-sim.onnx srmdnf_x4.param srmd_x2.bin
```
