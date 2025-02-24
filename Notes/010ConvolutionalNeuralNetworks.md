# Convolutional Neural Networks

## Why convolutional neural networks

之前我们学习的神经网络中的层叫做**全连接层 (fully-connected layer)**, 全连接层会**丧失输入的空间结构信息**，因为它会将输入特征都展开为一维向量然后再进行训练，不考虑输入特征的空间结构信息。

![fullyconnectedlayer](Images/fullyconnectedlayer.png)

为了训练出准确度更高的模型，我们需要卷积神经网络。

## Convolutional Layer

### 对于单个样本

一张图片可以表示为大小为 $3*32*32$ 的三维矩阵：

- 3表示图像的通道数，对应RGB (红色、绿色、蓝色) 三个颜色通道。因此图像的每个像素有三个值，分别表示红黄蓝的强度
- 32*32表示图像的高度和宽度，表明图像有32像素的高度和32像素的宽度

卷积核是一个包含可学习权重的小矩阵，它通过与输入数据的每个局部区域进行逐元素相乘并求和的方式来提取特征。在卷积神经网络中，卷积核通常会在输入图像上滑动，并在每个位置执行卷积操作. **卷积核的深度必须和通道数相同**。

在进行卷积操作时，卷积核会对输入数据进行滑动窗口操作，在相应的位置上进行element-wise乘积然后相加，得到**激活图 (Activation Map)**。

![filter](Images/filter.gif)

也可以使用多个卷积核，得到多个激活图，这样可以**进一步提取输入数据的特征**。

![multiconvokernels](Images/multiconvokernels.png)
