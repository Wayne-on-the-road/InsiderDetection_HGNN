import csv
import numpy as np
import pandas as pd
import torch
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

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,auc,roc_curve

# Install required packages.
import os
os.environ['TORCH'] = torch.__version__

#%% ## Creating a Heterogeneous Link-level GNN
from torch_geometric.nn import SAGEConv, to_hetero, GraphConv, GATConv
import torch.nn.functional as F

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


class Model_all(torch.nn.Module):
    def __init__(self, hidden_channels, GNN_model):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.UserDay_feature_lin = torch.nn.Linear(5, hidden_channels)
        self.UserDay_node_emb = torch.nn.Embedding(data['UserDay'].num_nodes, hidden_channels)

        self.supervisor_node_emb = torch.nn.Embedding(data['supervisor'].num_nodes, hidden_channels)
        self.user_node_emb = torch.nn.Embedding(data['user'].num_nodes, hidden_channels)

        self.output = torch.nn.Linear(hidden_channels, 2)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, GNN_model)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())


        # self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        #print(data['UserDay'].num_nodes, data['UserDay'].node_id)
        x_dict = {
          'UserDay': self.UserDay_feature_lin(data['UserDay'].x.float()) + self.UserDay_node_emb(data['UserDay'].node_id),
          'supervisor': self.supervisor_node_emb(data['supervisor'].node_id),
          'user': self.user_node_emb(data['user'].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        out = self.output(x_dict['UserDay'])
        out = F.log_softmax(out, dim=1)
        pred = out

        return pred, x_dict

class Model_supervison(torch.nn.Module):
    def __init__(self, hidden_channels, GNN_model):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.UserDay_feature_lin = torch.nn.Linear(5, hidden_channels)
        self.UserDay_node_emb = torch.nn.Embedding(data['UserDay'].num_nodes, hidden_channels)

        self.supervisor_node_emb = torch.nn.Embedding(data['supervisor'].num_nodes, hidden_channels)
        # self.user_node_emb = torch.nn.Embedding(data['user'].num_nodes, hidden_channels)

        self.output = torch.nn.Linear(hidden_channels, 2)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, GNN_model)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())


        # self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        #print(data['UserDay'].num_nodes, data['UserDay'].node_id)
        x_dict = {
          'UserDay': self.UserDay_feature_lin(data['UserDay'].x.float()) + self.UserDay_node_emb(data['UserDay'].node_id),
          'supervisor': self.supervisor_node_emb(data['supervisor'].node_id),
          # 'user': self.user_node_emb(data['user'].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        out = self.output(x_dict['UserDay'])
        out = F.log_softmax(out, dim=1)
        pred = out

        return pred, x_dict
class Model_SameUser(torch.nn.Module):
    def __init__(self, hidden_channels, GNN_model):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.UserDay_feature_lin = torch.nn.Linear(5, hidden_channels)
        self.UserDay_node_emb = torch.nn.Embedding(data['UserDay'].num_nodes, hidden_channels)

        # self.supervisor_node_emb = torch.nn.Embedding(data['supervisor'].num_nodes, hidden_channels)
        self.user_node_emb = torch.nn.Embedding(data['user'].num_nodes, hidden_channels)

        self.output = torch.nn.Linear(hidden_channels, 2)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, GNN_model)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())


        # self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        #print(data['UserDay'].num_nodes, data['UserDay'].node_id)
        x_dict = {
          'UserDay': self.UserDay_feature_lin(data['UserDay'].x.float()) + self.UserDay_node_emb(data['UserDay'].node_id),
          # 'supervisor': self.supervisor_node_emb(data['supervisor'].node_id),
          'user': self.user_node_emb(data['user'].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        out = self.output(x_dict['UserDay'])
        out = F.log_softmax(out, dim=1)
        pred = out

        return pred, x_dict

#%% ## Training a Heterogeneous GNN
def train(data):
    optimizer.zero_grad()
    data.to(device)
    pred,_ = model(data)

    ground_truth_train = data['UserDay'].y[data['UserDay'].train_mask]
    ground_truth_test = data['UserDay'].y[data['UserDay'].test_mask]

    pred_train = pred[data['UserDay'].train_mask]

    loss_train = loss_func(pred_train, ground_truth_train)
    loss_test = loss_func(pred[data['UserDay'].test_mask], ground_truth_test)

    loss_train.backward()
    optimizer.step()

    return loss_train.detach().cpu().item(), loss_test.detach().cpu().item()


#%% ablation setting
ablation_settings = ['All_relation']#,'Supervision_relation','SameUser_relation'] #'All','Title','Department','Manager','None'
GNN_models = ['GCN', 'GAT', 'GraphSAGE']
round = 30
batch_size= 2048
hidden_channel=16 #16,32,64,128
Epoch=400
graph_file_path = './data/insider_detection_heterogeneous_graph.pt'
# graph_file_path = './data/insider_detection_heterogeneous_graph_3relations.pt'
data_path = './CERT4.2/user_feature_label/data-wise_total.csv'
result_dir = './result_dir' # delta 0.0000001
# result_dir = './result_dir_delta-06' # delta 0.000001

loss_func = torch.nn.CrossEntropyLoss()
# csv file to store the final performance result for each model and feature combination:
os.makedirs(result_dir, exist_ok=True)
total_perform_file = os.path.join(result_dir, f'total_result.csv')
with open(total_perform_file, 'w', newline='') as f:
    writer = csv.writer(f)
    my_list = ['relation', 'GNN_model', 'round', 'total_epoch', 'classifier', 'features', 'Acc', 'Pre', 'Rec', 'F1','Auc']
    writer.writerow(my_list)


for ablation_setting in ablation_settings:
    result_dir_ablation = os.path.join(result_dir, ablation_setting)
    os.makedirs(result_dir_ablation, exist_ok=True)
    for GNN_model in GNN_models:
        result_dir_ablation_gnn = os.path.join(result_dir_ablation, GNN_model)
        os.makedirs(result_dir_ablation_gnn, exist_ok=True)
        best_model_dir = os.path.join(result_dir_ablation_gnn, 'early_stop_model')
        os.makedirs(best_model_dir, exist_ok=True)
        loss_trend_dir = os.path.join(result_dir_ablation_gnn, 'loss_trend')
        os.makedirs(loss_trend_dir, exist_ok=True)
        perform_file_dir = os.path.join(result_dir_ablation_gnn, 'perform_file')
        os.makedirs(perform_file_dir, exist_ok=True)
        for r in range(round):
            setting = f'{ablation_setting}-{GNN_model}-BS_{batch_size}-HC_{hidden_channel}-{r}th'

        ## Load the heterogeneous graph data
            data = torch.load(graph_file_path)
            print('Ablation_setting: ', ablation_setting)
            if ablation_setting == 'SameUser_relation':
                del data['supervisor']
                del data['UserDay', 'has_supervisor', 'supervisor']
                data = T.ToUndirected()(data)
                model = Model_SameUser(hidden_channel, GNN_model=GNN_model)
            elif ablation_setting == 'Supervision_relation':
                del data['user']
                del data['UserDay', 'has_SameUser', 'user']
                data = T.ToUndirected()(data)
                model = Model_supervison(hidden_channel, GNN_model=GNN_model)
            elif ablation_setting == 'All_relation':
                data = T.ToUndirected()(data)
                model = Model_all(hidden_channel, GNN_model=GNN_model)

            print(data)


        #%% ## early stop train process
            print(model)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Device: '{device}'")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # prepare file for best model archive

            best_model_path = os.path.join(best_model_dir, setting+'-best.pt')
            early_stopping = EarlyStopping(save_path=best_model_path, verbose=(True), patience=20, delta=0.000001, metric='loss')

            train_loss_list = []
            validation_loss_list = []
            total_start_time = time.time()
            for epoch in np.arange(Epoch):
                start_time = time.time()
                train_loss, validation_loss = train(data)
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Epoch: {epoch:03d}, train_time: {train_time} seconds, train_Loss: {train_loss:.4f},val_Loss: {validation_loss:.4f}")
                train_loss_list.append(train_loss)
                validation_loss_list.append(validation_loss)
                early_stopping(validation_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch:", epoch)
                    break
            total_end_time = time.time()
            total_train_time = total_end_time - total_start_time
            print(f"Setting: {setting}, total_train_epoch:  {epoch:03d}, total_train_time: {total_train_time} seconds.")

            #show and save result:
            early_stopping.draw_trend(train_loss_list, validation_loss_list,save_path=os.path.join(loss_trend_dir, setting + '-loss_trend'))
            csv_file_path = os.path.join(loss_trend_dir, setting + '-loss_data.csv') # save loss progress data for each round
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([setting, "total_train_epoch:", epoch, 'total_train_time',total_train_time])
                writer.writerow([setting,'train_loss_list'])
                writer.writerow([setting,'validation_loss_list'])
                writer.writerow(train_loss_list)
                writer.writerow(validation_loss_list)

            # use graph feature to do classification for this round:

            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            out,_ = model(data)
            # print(data['UserDay'].x[data['UserDay'].train_mask])
            graph_feature_train = pd.DataFrame(out[data['UserDay'].train_mask].detach().numpy())
            original_feature_train = pd.DataFrame(data['UserDay'].x[data['UserDay'].train_mask].detach().numpy())
            train_x = original_feature_train
            train_y = data['UserDay'].y[data['UserDay'].train_mask].detach().numpy()
            train_x_g = pd.concat([original_feature_train, graph_feature_train], axis=1)
            train_y_g = train_y

            graph_feature_test = pd.DataFrame(out[data['UserDay'].test_mask].detach().numpy())
            original_feature_test = pd.DataFrame(data['UserDay'].x[data['UserDay'].test_mask].detach().numpy())
            test_x = original_feature_test
            test_y = data['UserDay'].y[data['UserDay'].test_mask].detach().numpy()
            test_x_g = pd.concat([original_feature_test, graph_feature_test], axis=1)
            test_y_g = test_y


            # classifer test
            LR = LogisticRegression()
            RF = RandomForestClassifier()
            SVM = svm.SVC(probability=(True))
            GNB = GaussianNB()
            classifier_list = [RF, LR, GNB, SVM]
            #classifier_name_list = ["c_RF"]  # "c_LR", "c_RF", "c_SVM", "c_NN", "c_GNB"
            classifier_name_list = ["RF", "LR", "GNB", "SVM"]

            file_perform_per_round = os.path.join(perform_file_dir,
                                         setting + '-perform_data.npy')  #
            perform_per_round = {}
            for classifier,classifier_name in zip(classifier_list,classifier_name_list):
                # train and test with original feature
                metrics = report_metrics(classifier,train_x,train_y,test_x,test_y)
                metrics_g = report_metrics(classifier,train_x_g,train_y_g,test_x_g,test_y_g)

                # save the results
                with open(total_perform_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    result = [ablation_setting, GNN_model, r, epoch, classifier_name, 'original',
                               metrics['Acc'], metrics['Pre'], metrics['Rec'], metrics['F1'],metrics['Auc']]
                    result_g = [ablation_setting, GNN_model, r, epoch, classifier_name, 'original+graph',
                               metrics_g['Acc'], metrics_g['Pre'], metrics_g['Rec'], metrics_g['F1'],metrics_g['Auc']]
                    writer.writerow(result)
                    writer.writerow(result_g)

                print([result, result_g])

                perform_per_round[classifier_name] = {
                    'original': metrics,
                    'with_graph': metrics_g,
                }

            np.save(file_perform_per_round, perform_per_round)




