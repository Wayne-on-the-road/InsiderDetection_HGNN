import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
print(torch.__version__)
from early_stop_v1 import EarlyStopping
from tool import report_metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import tqdm
import torch.nn.functional as F
import time # wei: this is missing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from torch_geometric.nn import SAGEConv, to_hetero, GraphConv, GATConv

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,auc,roc_curve



class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, GNN_model):
        super().__init__()
        if GNN_model == 'GCN':
            self.conv1 = GraphConv(hidden_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
        elif GNN_model == 'GAT':
            self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops=False)
        elif GNN_model == 'GraphSAGE':
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class CNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(CNNModule, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # First convolutional layer
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # Second convolutional layer
            torch.nn.ReLU(),
            torch.nn.Flatten()  # Flatten the output for fully connected layers
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d (batch_size, 1, hidden_size)
        return self.cnn(x)


class Model_all(nn.Module):
    def __init__(self, data, hidden_channels, GNN_model, residual=True, use_gnn=True, classifier='MLP', use_node_emb=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.use_gnn = use_gnn
        self.classifier = classifier
        self.use_node_emb = use_node_emb

        self.UserDay_lin = nn.Linear(data['UserDay'].x.shape[1], hidden_channels)
        if self.use_node_emb:
            self.UserDay_node_emb = nn.Embedding(data['UserDay'].num_nodes, hidden_channels)
            self.supervisor_node_emb = nn.Embedding(data['supervisor'].num_nodes, hidden_channels)
            self.user_node_emb = nn.Embedding(data['user'].num_nodes, hidden_channels)

        model_input_size = hidden_channels

        # Classifier setup
        if self.classifier == 'MLP':
            self.model = nn.Sequential(
                nn.Linear(model_input_size, model_input_size // 2),
                nn.ReLU(),
                nn.Linear(model_input_size // 2, model_input_size // 2),
                nn.ReLU()
            )
            self.output = nn.Linear(model_input_size // 2, 2)
        elif self.classifier == 'CNN':
            self.model = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier == '2dCNN':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier is None:
            self.output = nn.Linear(model_input_size, 2)

        if self.use_gnn:
            self.gnn = GNN(hidden_channels, GNN_model)
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData) -> Tensor:
        UserDay_features = self.UserDay_lin(data['UserDay'].x.float())
        if self.use_node_emb:
            x_dict = {
                'UserDay': UserDay_features + self.UserDay_node_emb(data['UserDay'].node_id),
                'supervisor': self.supervisor_node_emb(data['supervisor'].node_id),
                'user': self.user_node_emb(data['user'].node_id),
            }
        else:
            dummy_supervisor_features = torch.zeros((data['supervisor'].num_nodes, self.hidden_channels), device=data['supervisor'].node_id.device)
            dummy_user_features = torch.zeros((data['user'].num_nodes, self.hidden_channels), device=data['user'].node_id.device)
            x_dict = {
                'UserDay': UserDay_features,
                'supervisor': dummy_supervisor_features,
                'user': dummy_user_features,
            }

        if self.use_gnn:
            x_dict_g = self.gnn(x_dict, data.edge_index_dict)
            out = x_dict_g['UserDay'] + x_dict['UserDay'] if self.residual else x_dict_g['UserDay']
        else:
            out = x_dict['UserDay']

        # Reshape based on classifier type
        if self.classifier == 'CNN':
            out = out.unsqueeze(1)  # For Conv1d
        elif self.classifier == '2dCNN':
            batch_size, feature_size = out.size(0), out.size(1)
            sqrt_feature_size = int(feature_size ** 0.5)
            out = out.view(batch_size, 1, sqrt_feature_size, sqrt_feature_size) if sqrt_feature_size * sqrt_feature_size == feature_size else out.view(batch_size, 1, feature_size // 2, 2)

        logits = self.output(self.model(out) if self.classifier else out)
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        return predicted_labels, probabilities

# Model_SameUser
class Model_SameUser(nn.Module):
    def __init__(self, data, hidden_channels, GNN_model, residual=True, use_gnn=True, classifier='MLP', use_node_emb=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.use_gnn = use_gnn
        self.classifier = classifier
        self.use_node_emb = use_node_emb

        self.UserDay_lin = nn.Linear(data['UserDay'].x.shape[1], hidden_channels)
        if self.use_node_emb:
            self.UserDay_node_emb = nn.Embedding(data['UserDay'].num_nodes, hidden_channels)
            self.user_node_emb = nn.Embedding(data['user'].num_nodes, hidden_channels)

        model_input_size = hidden_channels

        # Classifier setup
        if self.classifier == 'MLP':
            self.model = nn.Sequential(
                nn.Linear(model_input_size, model_input_size // 2),
                nn.ReLU(),
                nn.Linear(model_input_size // 2, model_input_size // 2),
                nn.ReLU()
            )
            self.output = nn.Linear(model_input_size // 2, 2)
        elif self.classifier == 'CNN':
            self.model = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier == '2dCNN':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier is None:
            self.output = nn.Linear(model_input_size, 2)

        if self.use_gnn:
            self.gnn = GNN(hidden_channels, GNN_model)
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData) -> Tensor:
        UserDay_features = self.UserDay_lin(data['UserDay'].x.float())
        if self.use_node_emb:
            x_dict = {
                'UserDay': UserDay_features + self.UserDay_node_emb(data['UserDay'].node_id),
                'user': self.user_node_emb(data['user'].node_id),
            }
        else:
            dummy_user_features = torch.zeros((data['user'].num_nodes, self.hidden_channels), device=data['user'].node_id.device)
            x_dict = {
                'UserDay': UserDay_features,
                'user': dummy_user_features,
            }

        if self.use_gnn:
            x_dict_g = self.gnn(x_dict, data.edge_index_dict)
            out = x_dict_g['UserDay'] + x_dict['UserDay'] if self.residual else x_dict_g['UserDay']
        else:
            out = x_dict['UserDay']

        # Reshape based on classifier type
        if self.classifier == 'CNN':
            out = out.unsqueeze(1)
        elif self.classifier == '2dCNN':
            batch_size, feature_size = out.size(0), out.size(1)
            sqrt_feature_size = int(feature_size ** 0.5)
            out = out.view(batch_size, 1, sqrt_feature_size, sqrt_feature_size) if sqrt_feature_size * sqrt_feature_size == feature_size else out.view(batch_size, 1, feature_size // 2, 2)

        logits = self.output(self.model(out) if self.classifier else out)
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        return predicted_labels, probabilities

# Model_Supervision
class Model_Supervision(nn.Module):
    def __init__(self, data, hidden_channels, GNN_model, residual=True, use_gnn=True, classifier='MLP', use_node_emb=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.use_gnn = use_gnn
        self.classifier = classifier
        self.use_node_emb = use_node_emb

        self.UserDay_lin = nn.Linear(data['UserDay'].x.shape[1], hidden_channels)
        if self.use_node_emb:
            self.UserDay_node_emb = nn.Embedding(data['UserDay'].num_nodes, hidden_channels)
            self.supervisor_node_emb = nn.Embedding(data['supervisor'].num_nodes, hidden_channels)

        model_input_size = hidden_channels

        # Classifier setup
        if self.classifier == 'MLP':
            self.model = nn.Sequential(
                nn.Linear(model_input_size, model_input_size // 2),
                nn.ReLU(),
                nn.Linear(model_input_size // 2, model_input_size // 2),
                nn.ReLU()
            )
            self.output = nn.Linear(model_input_size // 2, 2)
        elif self.classifier == 'CNN':
            self.model = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier == '2dCNN':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.output = nn.Linear(model_input_size * 32, 2)
        elif self.classifier is None:
            self.output = nn.Linear(model_input_size, 2)

        if self.use_gnn:
            self.gnn = GNN(hidden_channels, GNN_model)
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData) -> Tensor:
        UserDay_features = self.UserDay_lin(data['UserDay'].x.float())
        if self.use_node_emb:
            x_dict = {
                'UserDay': UserDay_features + self.UserDay_node_emb(data['UserDay'].node_id),
                'supervisor': self.supervisor_node_emb(data['supervisor'].node_id),
            }
        else:
            dummy_supervisor_features = torch.zeros((data['supervisor'].num_nodes, self.hidden_channels), device=data['supervisor'].node_id.device)
            x_dict = {
                'UserDay': UserDay_features,
                'supervisor': dummy_supervisor_features,
            }

        if self.use_gnn:
            x_dict_g = self.gnn(x_dict, data.edge_index_dict)
            out = x_dict_g['UserDay'] + x_dict['UserDay'] if self.residual else x_dict_g['UserDay']
        else:
            out = x_dict['UserDay']

        # Reshape based on classifier type
        if self.classifier == 'CNN':
            out = out.unsqueeze(1)
        elif self.classifier == '2dCNN':
            batch_size, feature_size = out.size(0), out.size(1)
            sqrt_feature_size = int(feature_size ** 0.5)
            out = out.view(batch_size, 1, sqrt_feature_size, sqrt_feature_size) if sqrt_feature_size * sqrt_feature_size == feature_size else out.view(batch_size, 1, feature_size // 2, 2)

        logits = self.output(self.model(out) if self.classifier else out)
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        return predicted_labels, probabilities