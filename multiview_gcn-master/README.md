# 多视图图卷积网络项目说明

## 项目概述
本项目实现了一个多视图图卷积网络（Multiview GCN），用于处理多视图图数据。项目包含数据处理、模型定义、模型训练和测试等功能。

## 文件说明
### 1. `data_processing.py`
- **功能**：处理图数据，将不同类型的节点特征和边信息整合为一个图数据对象。
- **使用方法**：在其他文件中导入`process_data`函数，传入节点特征和边信息，即可得到处理后的图数据。
- **输入**：
  - `lncRNA_features`：lncRNA节点的特征矩阵。
  - `disease_features`：疾病节点的特征矩阵。
  - `edges`：边信息列表。
- **输出**：一个`torch_geometric.data.Data`对象，包含整合后的节点特征和边信息。

### 2. `model_training.py`
#### 2.1 `MultiviewGCN`类
- **功能**：定义多视图图卷积网络模型。
- **使用方法**：实例化`MultiviewGCN`类，传入输入特征数量、隐藏层通道数和类别数量，然后调用`forward`方法进行前向传播。
- **输入**：
  - `num_features`：输入特征数量。
  - `hidden_channels`：隐藏层通道数。
  - `num_classes`：类别数量。
- **输出**：模型的预测结果。

#### 2.2 `MultiviewFeatureExtractor`类
- **功能**：提取多视图特征。
- **使用方法**：实例化`MultiviewFeatureExtractor`类，传入视图数量和特征数量，然后调用`forward`方法进行特征提取。
- **输入**：
  - `num_views`：视图数量。
  - `num_features`：特征数量。
- **输出**：提取后的多视图特征。

#### 2.3 `train_model`函数
- **功能**：训练多视图图卷积网络模型。
- **使用方法**：调用`train_model`函数，传入模型、图数据和训练轮数，即可开始训练。
- **输入**：
  - `model`：多视图图卷积网络模型。
  - `data`：图数据对象。
  - `epochs`：训练轮数，默认为200。
- **输出**：训练好的模型。

### 3. `test_multiview_gcn.py`
- **功能**：对多视图图卷积网络的各个组件进行单元测试。
- **使用方法**：运行该文件，即可执行所有测试用例。

## 示例使用
### 1. 数据处理
```python
from data_processing import process_data
import torch

# 示例数据
num_lncRNAs = 100
num_diseases = 50
num_features = 16
lncRNA_features = torch.randn(num_lncRNAs, num_features)
disease_features = torch.randn(num_diseases, num_features)
edges = [[i, j + num_lncRNAs] for i in range(num_lncRNAs) for j in range(num_diseases)]

# 处理数据
data = process_data(lncRNA_features, disease_features, edges)
```

### 2. 模型定义和训练
```python
from model_training import MultiviewGCN, MultiviewFeatureExtractor, train_model

# 创建模型
model = MultiviewGCN(num_features * 3, 32, 2)
extractor = MultiviewFeatureExtractor(3, num_features)

# 提取特征
data.x = extractor(data.x)

# 训练模型
trained_model = train_model(model, data, epochs=1)
```

### 3. 单元测试
```python
python test_multiview_gcn.py
```

## 注意事项
- 请确保已经安装了`torch`和`torch_geometric`库。
- 在训练模型时，可以根据需要调整训练轮数和其他超参数。