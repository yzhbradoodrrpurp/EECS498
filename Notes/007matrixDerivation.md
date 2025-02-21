# 矩阵求导

## 常见矩阵求导公式

### 标量函数

- $f(\boldsymbol{x}) = \boldsymbol{a}^T \boldsymbol{x}$ ，其中 $\boldsymbol{a}, \boldsymbol{x}$ 都是 $n * 1$ 的列向量，则 $\frac{\partial{f}}{\partial{\boldsymbol{x}}} = \boldsymbol{a}$
- $f(\boldsymbol{x}) = \boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x}$ ，其中 $\boldsymbol{x}$ 是 $n * 1$ 的列向量, $\boldsymbol{A}$ 是 $n * n$ 的矩阵，则 $\frac{\partial{f}}{\partial{\boldsymbol{x}}} = 2 \boldsymbol{A} \boldsymbol{x}$

### 向量函数

$f(\boldsymbol x) = \boldsymbol{A} \boldsymbol{x}$ ，其中 $\boldsymbol A$ 是 $m * n$ 的矩阵, $\boldsymbol x$ 是 $n * 1$ 的向量，则 $\frac{\partial f}{\partial \boldsymbol x} = \boldsymbol A$

## 推导SVM Loss的梯度

### 对于单个样本

假设 $\boldsymbol x_i$ , $\boldsymbol W$ , $\boldsymbol y_i$ :

- $\boldsymbol x_i$ 的形状为 $1 * n$ 
- $\boldsymbol W$ 的形状为 $n * c$ 

- $\boldsymbol y_i$ 是标量，表示 $\boldsymbol x_i$ 的正确类别

有Loss函数: $L_i = \sum_{j \neq y_i}{max(0, s_j - s_{y_i}) + \Delta}$

- $s_j$ 为样本分入第j类的分数，可写作 $\boldsymbol x_i \boldsymbol W_j$
- $s_{y_i}$ 为样本分入正确类别的分数，可写作 $\boldsymbol x_i \boldsymbol W_{y_i}$
- $\Delta$ 是间隔参数

那么对于 $\boldsymbol W_j$ 的梯度，

- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta > 0$ ，则 $\frac{\partial L_i}{\partial \boldsymbol W_j} = \boldsymbol x_i^T$
- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta \le 0$ ，则 $\frac{\partial L_i}{\partial \boldsymbol W_j} = 0$

那么对于 $W_{y_i}$ 的梯度, $\frac{\partial L_i}{\partial \boldsymbol W_{y_i}} = \sum_{j \neq y_i}{- \boldsymbol x_i^T} \cdot \boldsymbol{1} {(s_j - s_{y_i} + \Delta > 0)}$

- $\boldsymbol{1} {(s_j - s_{y_i} + \Delta > 0)}$ 是指示函数，表明括号中条件成立时，值为1；不成立时，值为0

### 在整个训练集上

将所有样本的梯度相加并取平均值，就得到了在整个训练集上的梯度: $\frac{\partial L}{\partial \boldsymbol W} = \frac{1}{N} \sum_{i=1}^{N}{\frac{\partial L_i}{\partial \boldsymbol W}}$

## 推导Cross-Entropy Loss梯度

### 对于单个样本

假设 $\boldsymbol x_i$ , $\boldsymbol W$ , $\boldsymbol y_i$ :

- $\boldsymbol x_i$ 的形状为 $1 * n$ 
- $\boldsymbol W$ 的形状为 $n * c$ 

- $\boldsymbol y_i$ 是标量，表示 $\boldsymbol x_i$ 的正确类别

有Loss函数: $L_i = -\log{p_{y_i}}$

- $p_j$ 是第j个类别分数softmax概率化后的数值, $p_{y_i}$ 是正确类别分数的softmax概率化后的数值，可以写作 $\frac{e^{s_{y_i}}}{\sum{e^{s_j}}}$
  - $s_j$ 为样本分入第j类的分数，可写作 $\boldsymbol x_i \boldsymbol W_j$
  - $s_{y_i}$ 为样本分入正确类别的分数，可写作 $\boldsymbol x_i \boldsymbol W_{y_i}$

那么对于 $\boldsymbol W_j$ 的梯度，可以用链式法则，

- $\frac{\partial L_i}{\partial s_j} = p_j - \boldsymbol{1}{(j=y_i)}$

> Softmax函数的概率 $p_j$ 定义为: $p_j = \frac{e^{s_j}}{\sum_{k=1}^{c}{e^{s_k}}}$
>
> Softmax函数有两个重要性质:
>
> - $\frac{\partial p_j}{\partial s_j} = p_j (1 - p_j)$
> - $\frac{\partial p_k}{\partial s_j} = - p_k p_j$ , $j \neq k$
>
> $\frac{\partial L_i}{\partial s_j} = \frac{\partial (-\log{p_{y_i}})}{\partial s_j} = - \frac{1}{p_{y_i}} \cdot \frac{\partial p_{y_i}}{\partial s_j}$
>
> 可以分成两种情况讨论，
>
> - $j = y_i$ 时: $\frac{\partial L_i}{\partial s_j} = - \frac{1}{p_{y_i}} \cdot p_j (1 - p_j) = p_{y_i} - 1$
> - $j \neq y_i$ 时: $\frac{\partial L_i}{\partial s_j} = - \frac{1}{p_{y_i}} \cdot (- p_{y_i} p_j) = p_j$

- $\frac{\partial s_j}{\partial \boldsymbol W_j} = \boldsymbol x_i^T$
- 则 $\frac{\partial L_i}{\partial \boldsymbol W_j} = (p_j - \boldsymbol{1}{(j=y_i)}) \cdot x_i^T$

那么对于 $W_{y_i}$ 的梯度，可以表示为 $\frac{\partial L_i}{\partial \boldsymbol W_{y_i}} = (p_{y_i} - 1) \cdot x_i^T$

### 在整个训练集上

将所有样本的梯度相加并取平均值，就得到了在整个训练集上的梯度: $\frac{\partial L}{\partial \boldsymbol W} = \frac{1}{N} \sum_{i=1}^{N}{\frac{\partial L_i}{\partial \boldsymbol W}}$
