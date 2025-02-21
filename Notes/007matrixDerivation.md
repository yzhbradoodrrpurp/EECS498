# 矩阵求导

## 常见矩阵求导公式

### 标量函数

- $f(\boldsymbol{x}) = \boldsymbol{a}^T \boldsymbol{x}$ ，其中 $\boldsymbol{a}, \boldsymbol{x}$ 都是 $n * 1$ 的列向量，则 $\frac{\partial{f}}{\partial{\boldsymbol{x}}} = \boldsymbol{a}$
- $f(\boldsymbol{x}) = \boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x}$ ，其中 $\boldsymbol{x}$ 是 $n * 1$ 的列向量，$\boldsymbol{A}$是 $n * n$ 的矩阵，则 $\frac{\partial{f}}{\partial{\boldsymbol{x}}} = 2 \boldsymbol{A} \boldsymbol{x}$

### 向量函数

$f(\boldsymbol x) = \boldsymbol{A} \boldsymbol{x}$ ，其中 $\boldsymbol A$ 是 $m * n$ 的矩阵，$\boldsymbol x$是 $n * 1$ 的向量，则 $\frac{\partial f}{\partial \boldsymbol x} = \boldsymbol A$

## 推导SVM Loss的梯度

### 对于单个样本

假设 $\boldsymbol x_i$ , $\boldsymbol W$ , $\boldsymbol y_i$ :

- $\boldsymbol x$ 的形状为 $1 * n$ 
- $\boldsymbol W$ 的形状为 $n * c$ 

- $\boldsymbol y_i$ 是标量，表示 $\boldsymbol x_i$ 的正确类别

有Loss函数: $L_i = \sum_{j \neq y_i}{max(0, s_j - s_{y_i}) + \Delta}$

- $s_j$ 为样本分入第j类的分数，可写作 $\boldsymbol x_i \boldsymbol W_j$
- $s_{y_i}$ 为样本分入正确类别的分数，可写作 $\boldsymbol x_i \boldsymbol W_{y_i}$
- $\Delta$ 是间隔参数

那么对于 $\boldsymbol W_j$ 的梯度，

- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta > 0$ ，则 $\frac{\partial L_i}{\partial \boldsymbol W_j} = \boldsymbol x^T$
- 如果 $j \neq y_i$ 且 $s_j - s_{y_i} + \Delta \le 0$ ，则 $\frac{\partial L_i}{\partial \boldsymbol W_j} = 0$

那么对于 $W_{y_i}$ 的梯度，$\frac{\partial L_i}{\partial \boldsymbol W_{y_i}} = \sum_{j \neq y_i}{-\boldsymbol x^T} \cdot \boldsymbol{1} {(s_j - s_{y_i} + \Delta > 0)}$

- $\boldsymbol{1} {(s_j - s_{y_i} + \Delta > 0)}$ 是指示函数，表明括号中条件成立时，值为1；不成立时，值为0

### 在整个训练集上

