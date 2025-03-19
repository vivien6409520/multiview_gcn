import torch
from torch_geometric.data import Data

# 构建lncRNAs和疾病的异质图
class HeterogeneousGraphBuilder:
    def __init__(self, lncRNA_features, disease_features, edges):
        self.lncRNA_features = lncRNA_features
        self.disease_features = disease_features
        self.edges = edges

    def build_graph(self):
        x = torch.cat([self.lncRNA_features, self.disease_features], dim=0)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        return x, edge_index

# 数据处理函数
def process_data(lncRNA_features, disease_features, edges):
    builder = HeterogeneousGraphBuilder(lncRNA_features, disease_features, edges)
    x, edge_index = builder.build_graph()
    data = Data(x=x, edge_index=edge_index)
    return data