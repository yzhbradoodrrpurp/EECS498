# 矩阵求导

## 常见矩阵求导公式

### 标量函数

- $f(\bold{x}) = \bold{a}^T \bold{x}$ ，其中 $\bold{a}, \bold{x}$ 都是 $n * 1$ 的列向量，则 $\frac{\part{f}}{\part{\bold{x}}} = \bold{a}$
- $f(\bold{x}) = \bold{x}^T \bold{A} \bold{x}$ ，其中 $\bold{x}$ 是 $n * 1$ 的列向量，$\bold{A}$ 是 $n * n$ 的矩阵，则 $\frac{\part{f}}{\part{\bold{x}}} = 2 \bold{A} \bold{x}$

### 向量函数

$f(\bold x) = \bold{A} \bold{x}$ ，其中 $\bold A$ 是 $m * n$ 的矩阵，$\bold x$ 是 $n * 1$ 的向量，则 $\frac{\part f}{\part \bold x} = \bold A$

## 推导SVM Loss的梯度

### 对于单个样本

假设 $\bold x_i$ , $\bold W$ , $\bold y_i$ :

- $\bold x$ 的形状为 $1 * n$ 
- $\bold W$ 的形状为 $n * c$ 

- $\bold y_i$ 是标量，表示 $\bold x_i$ 的正确类别

有Loss函数: $L_i = \sum_{j \neq y_i}{max(0, s_j - s_{y_i}) + \Delta}$

- $s_j$ 为样本分入第j类的分数，可写作 $\bold x_i \bold W_j$
- $s_{y_i}$ 为样本分入正确类别的分数，可写作 $\bold x_i \bold W_{y_i}$
- $\Delta$ 是间隔参数

那么对于 $\bold W_j$ 的梯度，

- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta > 0$ ，则 $\frac{\part L_i}{\part \bold W_j} = \bold x^T$
- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta \le 0$ ，则 $\frac{\part L_i}{\part \bold W_j} = 0$

那么对于 $W_{y_i}$ 的梯度，$\frac{\part L_i}{\part \bold W_{y_i}} = \sum_{j \neq y_i}{-\bold x^T} \cdot \bold{1}_{(s_j - s_{y_i} + \Delta > 0)}$

- $\bold{1}_{(s_j - s_{y_i} + \Delta > 0)}$ 是指示函数，表明括号中条件成立时，值为1；不成立时，值为0

### 在整个训练集上

