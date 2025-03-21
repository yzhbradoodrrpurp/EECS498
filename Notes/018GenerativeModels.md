# Generative Models

## Supervised vs Unsupervised Learning

监督学习本质上可以看作这样一个过程：需要训练的模型是 $f$，数据是 $(x, y)$，其中 $x$ 是输入数据 $y$ 是标签，通过梯度下降的方式达到 $f: x \rightarrow y$。之前学到的内容 (分类、回归、物体检测、图像描述等等) 都是属于监督学习。

而无监督学习就只有输入 $x$，没有标签 $y$，它的目标是学习到 $x$ 中的隐藏结构。一些无监督学习的例子有 K-Means Clustering, Principle Component Analysis, Autoencoder, Density Estimation 等等。

## Discriminative vs Generative Models

Discriminative Model 是在有限的标签 $y$ 中寻找最符合输入 $x$ 的过程。它的问题在于不能处理不合理的输入，所有标签的概率和为 1，总会有概率最大的标签出现，即便这个标签是错的。比如在一个只有猫、狗两个标签的模型中输入一个猴子，这个模型输出的标签无论怎样都不是猴子，它的输出只能是猫或者狗。

Generative Model 的目标是学习关于输入 $x$ 本身的概率分布，而不受到标签 $y$ 的影响。它是在整个关于输入 $x$ 的分布中进行学习，也就是说，它可以拒绝不合理的输入。

Conditional Generative Model 结合了 Discriminative Model 和 Generative Model，它学习的给定标签 $y$，应该得到怎样的输入 $x$。

Discriminative Model 可以看作是 $P(y|x)$，Generative Model 可以看作是 $P(x)$，Conditional Generative Model 可以看作是 $P(x|y)$，那么根据贝叶斯公式就可以得到 $P(x|y) = \frac{P(y|x)}{P(y)} P(x)$。也就是说，一个 Discriminative Model 可以通过某种转换得到 Conditional Generative Model。

![bayes](Images/bayes.png)

这是一个关于 Generative Model 的分类，后面将会讲到 Autoregressive Model, Variational Autoencoder 和 Generative Adversarial Networks。

![taxonomygenerativemodels](Images/taxonomygenerativemodels.png)

## Autoregressive Models



## Variational Autoencoder



## Generative Adversarial Networks