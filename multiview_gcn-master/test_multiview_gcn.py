import torch
from torch_geometric.data import Data
from data_processing import process_data
from model_training import MultiviewGCN, MultiviewFeatureExtractor, train_model
import unittest

class TestMultiviewGCN(unittest.TestCase):
    def setUp(self):
        # 示例数据
        self.num_lncRNAs = 100
        self.num_diseases = 50
        self.num_features = 16
        self.num_views = 3
        self.num_classes = 2

        self.lncRNA_features = torch.randn(self.num_lncRNAs, self.num_features)
        self.disease_features = torch.randn(self.num_diseases, self.num_features)
        self.edges = [[i, j + self.num_lncRNAs] for i in range(self.num_lncRNAs) for j in range(self.num_diseases)]
        self.lncRNA_labels = torch.randint(0, self.num_classes, (self.num_lncRNAs,))
        self.disease_labels = torch.randint(0, self.num_classes, (self.num_diseases,))

    def test_process_data(self):
        data = process_data(self.lncRNA_features, self.disease_features, self.edges)
        self.assertEqual(data.x.shape[0], self.num_lncRNAs + self.num_diseases)
        self.assertEqual(data.edge_index.shape[1], len(self.edges))

    def test_MultiviewGCN(self):
        model = MultiviewGCN(self.num_features * self.num_views, 32, self.num_classes)
        extractor = MultiviewFeatureExtractor(self.num_views, self.num_features)
        data = process_data(self.lncRNA_features, self.disease_features, self.edges)
        data.x = extractor(data.x)
        out = model(data.x, data.edge_index)
        self.assertEqual(out.shape[0], data.x.shape[0])
        self.assertEqual(out.shape[1], self.num_classes)

    def test_train_model(self):
        model = MultiviewGCN(self.num_features * self.num_views, 32, self.num_classes)
        extractor = MultiviewFeatureExtractor(self.num_views, self.num_features)
        data = process_data(self.lncRNA_features, self.disease_features, self.edges)
        data.x = extractor(data.x)
        data.y = torch.cat([self.lncRNA_labels, self.disease_labels], dim=0)
        trained_model = train_model(model, data, epochs=1)
        self.assertEqual(isinstance(trained_model, MultiviewGCN), True)

if __name__ == '__main__':
    unittest.main()