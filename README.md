# multiview_gcn

Project Overview
This project implements a ​Multi-View Graph Convolutional Network (MViGCN) to handle multi-view graph data. The system includes modules for data processing, model definition, training, and evaluation.

​File Structure
​**data_processing.py**
​Functionality: Processes graph data by integrating node features and edge information across multiple views into a unified graph data object.
​Usage:
python
from data_processing import process_data
# Example usage
data = process_data(lncRNA_features, disease_features, edges)
​Inputs:
lncRNA_features: Feature matrix for lncRNA nodes (shape: [num_lncRNAs, num_features]).
disease_features: Feature matrix for disease nodes (shape: [num_diseases, num_features]).
edges: List of edge connections (e.g., [i, j] indicates an edge between node i and j).
​Output:
A torch_geometric.data.Data object containing integrated node features and edge indices.
​**model_training.py**
​2.1 MultiviewGCN Class

​Functionality: Defines the MVGCN model for multi-view graph data.
​Usage:
python
model = MultiviewGCN(num_features, hidden_channels, num_classes)
predictions = model(data)
​Parameters:
num_features: Input feature dimension per node.
hidden_channels: Number of channels in hidden GCN layers.
num_classes: Number of prediction classes (e.g., disease types).
​Output: Predicted class probabilities for nodes.
​2.2 MultiviewFeatureExtractor Class

​Functionality: Extracts multi-view features using graph convolutional layers.
​Usage:
python
extractor = MultiviewFeatureExtractor(num_views, num_features)
extracted_features = extractor(data)
​Parameters:
num_views: Number of input views (e.g., lncRNA, disease, miRNA).
num_features: Feature dimension per view.
​Output: Concatenated multi-view feature vectors for nodes.
​2.3 train_model Function

​Functionality: Trains the MVGCN model using graph data.
​Usage:
python
trained_model = train_model(model, data, epochs=100)
​Parameters:
model: MVGCN instance.
data: Graph data object.
epochs: Number of training epochs (default: 200).
​Output: Trained model instance.
​**test_multiview_gcn.py**
​Functionality: Runs unit tests for all components of the MVGCN framework.
​Usage:
bash
python test_multiview_gcn.py
​Example Usage
1. Data Processing
python
from data_processing import process_data
import torch

# Generate synthetic data
num_lncRNAs = 100
num_diseases = 50
num_features = 16
lncRNA_features = torch.randn(num_lncRNAs, num_features)  # lncRNA node features
disease_features = torch.randn(num_diseases, num_features)  # Disease node features
edges = [[i, j + num_lncRNAs] for i in range(num_lncRNAs) for j in range(num_diseases)]  # Edges between lncRNAs and diseases

# Process data into a graph object
data = process_data(lncRNA_features, disease_features, edges)
2. Model Definition and Training
python
from model_training import MultiviewGCN, MultiviewFeatureExtractor, train_model

# Initialize model and feature extractor
model = MultiviewGCN(num_features * 3, 32, 2)  # Input features: 3 views × 16 features
extractor = MultiviewFeatureExtractor(3, num_features)  # 3 views, 16 features per view

# Extract multi-view features
data.x = extractor(data.x)  # Update node features with extracted multi-view representations

# Train the model
trained_model = train_model(model, data, epochs=1)
3. Unit Testing
bash
python test_multiview_gcn.py
​Instructions
​Prerequisites:
Install required libraries:
bash
pip install torch torch_geometric
​Hyperparameter Tuning:
Adjust hidden_channels, num_classes, and epochs in model_training.py to optimize performance.
​Data Customization:
Modify the synthetic data generation in the example to match your dataset’s structure (e.g., node types, edge definitions).
​Notes
The MVGCN architecture leverages ​multi-view feature fusion and ​graph convolutional layers to capture complex relationships in heterogeneous biological networks.
For deployment, use the trained model to predict associations (e.g., lncRNA-disease links) via model.predict(data).
