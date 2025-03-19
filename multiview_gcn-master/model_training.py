import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from data_processing import process_data

# 多视图图卷积网络模型
class MultiviewGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(MultiviewGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 多视图特征提取模块
class MultiviewFeatureExtractor(torch.nn.Module):
    def __init__(self, num_views, num_features):
        super(MultiviewFeatureExtractor, self).__init__()
        self.views = torch.nn.ModuleList([
            torch.nn.Linear(num_features, num_features) for _ in range(num_views)
        ])

    def forward(self, x):
        return torch.cat([view(x) for view in self.views], dim=1)

# 模型训练函数


def train_model(model, data, epochs=200):
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
    return model

# 示例使用
if __name__ == '__main__':
    # 示例数据
    num_lncRNAs = 100
    num_diseases = 50
    num_features = 16
    num_views = 3
    num_classes = 2

    lncRNA_features = torch.randn(num_lncRNAs, num_features)
    disease_features = torch.randn(num_diseases, num_features)
    edges = [[i, j + num_lncRNAs] for i in range(num_lncRNAs) for j in range(num_diseases)]
    data = process_data(lncRNA_features, disease_features, edges)

    # 创建模型
    model = MultiviewGCN(num_features * num_views, 32, num_classes)
    extractor = MultiviewFeatureExtractor(num_views, num_features)
    data.x = extractor(data.x)

    # 训练模型
    trained_model = train_model(model, data)