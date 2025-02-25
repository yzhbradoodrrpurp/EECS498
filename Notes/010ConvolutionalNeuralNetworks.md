# Convolutional Neural Networks

## Why convolutional neural networks

之前我们学习的神经网络中的层叫做**全连接层 (fully-connected layer)**, 全连接层会**丧失输入的空间结构信息**，因为它会将输入特征都展开为一维向量然后再进行训练，不考虑输入特征的空间结构信息。

![fullyconnectedlayer](Images/fullyconnectedlayer.png)

为了训练出准确度更高的模型，我们需要卷积神经网络。

## Convolutional Layer

### 单个样本单个卷积层

一张图片可以表示为大小为3\*32\*32的三维矩阵：

- 3表示图像的通道数，对应RGB (红色、绿色、蓝色) 三个颜色通道。因此图像的每个像素有三个值，分别表示红黄蓝的强度
- 32*32表示图像的高度和宽度，表明图像有32像素的高度和32像素的宽度

**卷积核是一个包含可学习权重的小矩阵，它通过与输入数据的每个局部区域进行逐元素相乘并求和的方式来提取特征**。在卷积神经网络中，卷积核通常会在输入图像上滑动，并在每个位置执行卷积操作. **卷积核的深度必须和通道数相同**。

在进行卷积操作时，卷积核会对输入数据进行滑动窗口操作，在相应的位置上进行element-wise乘积然后相加，得到**激活图 (Activation Map)**。

![filter](Images/filter.gif)

也可以使用多个卷积核，得到多个激活图，这样可以**进一步提取输入数据的特征**。

![multiconvokernels](Images/multiconvokernels.png)

### 多个样本

输入数据可以是多个样本，组成一个大小为 $N * C_{in} * W * H$ 的四维矩阵，

- $N$ : 样本的个数
- $C_{in}$ :  单个样本的通道数
- $W, H$ : 表示单个样本的宽度和高度

卷积核也可以组成一个大小为 $C_{out} * C_{in} * K_w * K_h$ 的四维矩阵，

- $C_{out}$ : 卷积核的个数
- $C_{in}$ : 卷积核的深度，等于单个样本的通道数

- $K_w, K_h$ : 卷积核的宽度和高度

得到的激活图也是一个四维矩阵，大小为 $N * C_{out} * H' * W'$ 。

![batchcnnlayer](Images/batchcnnlayer.png)

### 多个卷积层

多个卷积核组成了一个卷积层，可以使用多个卷积层提取输入数据中的信息，另外别忘了使用激活函数改变线性结构。

![multiconvolayers](Images/multiconvolayers.png)

## 改进方法

### Problem 1: Shrinking Size of Hidden Layers

每经过一次卷积层，隐藏层的宽度和高度会越来越小，这样就会限制住卷积的次数。

> [见上图](# 多个卷积层): $N \times 3 \times 32 \times 32 \rightarrow N \times 6 \times 28 \times 28 \rightarrow N \times 26 \times 26$ , 每一层的宽度和高度越来越小。

### Solution 1: Padding
