# Main Process - Modified Residual Option
import csv
import os
from datetime import datetime
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.model_selection import KFold
from early_stop_v1 import EarlyStopping
import time
from No_sequence_models_EndToEnd_K_fold import Model_all, Model_Supervision, Model_SameUser
from tool_EndToEnd_KFolds import draw_roc, report_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def train(data, model, optimizer, loss_func, device, k_folds=3):
    model.train()

    # Filter out the training data based on the train mask
    train_idx = data['UserDay'].train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    X_train = data['UserDay'].x[train_idx]  # Train features
    y_train = data['UserDay'].y[train_idx]  # Train labels

    # Prepare K-Fold cross-validator
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Variables to accumulate metrics and losses
    total_train_loss = 0.0
    total_validation_loss = 0.0
    all_train_metrics = []
    all_validation_metrics = []

    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train)):
        # print(f"Fold {fold + 1}/{k_folds}")

        # Get the actual train/validation indices for the full data (not just the train split)
        train_fold = train_idx[train_fold_idx]
        val_fold = train_idx[val_fold_idx]

        # Create train and validation masks for the current fold
        train_mask = torch.zeros(data['UserDay'].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data['UserDay'].num_nodes, dtype=torch.bool)

        train_mask[train_fold] = True
        val_mask[val_fold] = True

        # Training step
        optimizer.zero_grad()

        # Forward pass through the model (getting both predicted labels and probabilities)
        pred_train_class, pred_train_proba = model(data)

        # Ground truth for training
        ground_truth_train = data['UserDay'].y[train_mask].cpu().numpy()

        # Calculate training loss (use the probabilities for calculating the loss)
        loss_train = loss_func(pred_train_proba[train_mask], torch.tensor(ground_truth_train).to(device))
        total_train_loss += loss_train.item()

        # Backpropagation
        loss_train.backward()
        optimizer.step()

        # Convert predictions and calculate metrics for training
        pred_train_class = pred_train_class[train_mask].detach().cpu().numpy()  # Predicted labels (0/1)
        pred_train_proba = pred_train_proba[train_mask][:, 1].detach().cpu().numpy()  # Probabilities for class 1
        train_metrics = report_metrics(ground_truth_train, pred_train_class, pred_train_proba, mode='train')
        all_train_metrics.append(train_metrics)

        # Validation step
        model.eval()
        with torch.no_grad():
            # Forward pass for validation
            pred_val_class, pred_val_proba = model(data)

            # Ground truth for validation
            ground_truth_val = data['UserDay'].y[val_mask].cpu().numpy()

            # Calculate validation loss
            loss_val = loss_func(pred_val_proba[val_mask], torch.tensor(ground_truth_val).to(device))
            total_validation_loss += loss_val.item()

            # Convert predictions and calculate metrics for validation
            pred_val_class = pred_val_class[val_mask].detach().cpu().numpy()  # Predicted labels (0/1)
            pred_val_proba = pred_val_proba[val_mask][:, 1].detach().cpu().numpy()  # Probabilities for class 1
            val_metrics = report_metrics(ground_truth_val, pred_val_class, pred_val_proba, mode='train')
            all_validation_metrics.append(val_metrics)

        model.train()  # Switch back to training mode

    # Average losses and metrics across all folds
    avg_train_loss = total_train_loss / k_folds
    avg_val_loss = total_validation_loss / k_folds

    # Average train and validation metrics
    avg_train_metrics = {key: np.mean([fold[key] for fold in all_train_metrics]) for key in all_train_metrics[0].keys()}
    avg_val_metrics = {key: np.mean([fold[key] for fold in all_validation_metrics]) for key in
                       all_validation_metrics[0].keys()}

    return avg_train_loss, avg_val_loss, avg_train_metrics, avg_val_metrics

# Basic settings
print_epoch = False
print_trend = False
save_loss = False
round = 30
batch_size = 2048
Epoch = 1000
use_gnn = True
use_node_emb = False

# ablation setting
hidden_channel = 16
residual = True  # Use True for "add" behavior, False for "None" behavior

classifiers = [None]# 'MLP','CNN','2dCNN',None
relation_settings = ['All_relation','Supervision_relation', 'SameUser_relation'] # ,'Supervision_relation', 'SameUser_relation'
GNN_models = ['GCN','GAT','GraphSAGE'] # ,'GAT','GraphSAGE'

graph_file_path = './data/insider_detection_heterogeneous_graph.pt'
data_path = './data/data-total.csv'

for classifier in classifiers:
    print("classifier: ", classifier)

    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    result_dir = f"result_NoSeq_e5_emb/" # {date_time}
    loss_func = torch.nn.CrossEntropyLoss()

    # Create directory to store results
    os.makedirs(result_dir, exist_ok=True)
    grand_setting = f'Res_{residual}-HC_{hidden_channel}-GNN_{use_gnn}-Classifier_{classifier}-NodeEmb_{use_node_emb}'
    total_perform_file = os.path.join(result_dir, f'total_result-{grand_setting}.csv')
    with open(total_perform_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['relation', 'GNN_model', 'round', 'total_epoch', 'Acc', 'Pre', 'Rec', 'F1', 'Auc'])

    for relation_setting in relation_settings:
        result_dir_ablation = os.path.join(result_dir, relation_setting)
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
                setting = f'{grand_setting}-{relation_setting}-{GNN_model}-{r}th'
                trend_fig_path = os.path.join(loss_trend_dir, setting + '-loss_trend')

                # Load the heterogeneous graph data
                data = torch.load(graph_file_path)
                print('relation_setting: ', relation_setting)
                if relation_setting == 'SameUser_relation':
                    del data['supervisor']
                    del data['UserDay', 'has_supervisor', 'supervisor']
                    data = T.ToUndirected()(data)
                    model = Model_SameUser(data=data, hidden_channels=hidden_channel, GNN_model=GNN_model, residual=residual, use_gnn=use_gnn, classifier=classifier, use_node_emb=use_node_emb)
                elif relation_setting == 'Supervision_relation':
                    del data['user']
                    del data['UserDay', 'has_SameUser', 'user']
                    data = T.ToUndirected()(data)
                    model = Model_Supervision(data=data, hidden_channels=hidden_channel, GNN_model=GNN_model, residual=residual, use_gnn=use_gnn, classifier=classifier, use_node_emb=use_node_emb)
                elif relation_setting == 'All_relation':
                    data = T.ToUndirected()(data)
                    model = Model_all(data=data, hidden_channels=hidden_channel, GNN_model=GNN_model, residual=residual, use_gnn=use_gnn, classifier=classifier, use_node_emb=use_node_emb)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Device: '{device}'")
                model = model.to(device)
                data.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                best_model_path = os.path.join(best_model_dir, setting + '-best.pt')
                early_stopping = EarlyStopping(save_path=best_model_path, patience=10, delta=0.000001, metric='loss')

                train_loss_list = []
                validation_loss_list = []
                total_start_time = time.time()
                for epoch in np.arange(Epoch):
                    start_time = time.time()

                    # Apply the modified train function with k-fold CV
                    train_loss, validation_loss, train_metrics, test_metrics = train(data, model, optimizer, loss_func, device)

                    end_time = time.time()
                    train_time = end_time - start_time
                    if print_epoch:
                        print(
                            f"Epoch: {epoch:03d}, train_time: {train_time:.4f} seconds, train_Loss: {train_loss:.4f}, val_Loss: {validation_loss:.4f}")
                    train_loss_list.append(train_loss)
                    validation_loss_list.append(validation_loss)
                    early_stopping(validation_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping at epoch:", epoch)
                        break
                total_end_time = time.time()
                total_train_time = total_end_time - total_start_time
                print(f"Setting: {setting}, total_train_epoch:  {epoch:03d}, total_train_time: {total_train_time} seconds.")

                # Load and evaluate the best model on the test set
                model.load_state_dict(torch.load(best_model_path))
                model.eval()
                with torch.no_grad():
                    pred_test_class, pred_test_proba = model(data)
                    pred_test = pred_test_class[data['UserDay'].test_mask]
                    ground_truth_test = data['UserDay'].y[data['UserDay'].test_mask].cpu().numpy()
                    loss_test = loss_func(pred_test_proba[data['UserDay'].test_mask], torch.tensor(ground_truth_test).to(device))

                    pred_test_proba = pred_test_proba[data['UserDay'].test_mask][:, 1].detach().cpu().numpy()
                    pred_test_class = pred_test_class[data['UserDay'].test_mask].detach().cpu().numpy()

                    metrics = report_metrics(ground_truth_test, pred_test_class, pred_test_proba, mode='test')
                    print(metrics['Acc'], metrics['Pre'], metrics['Rec'], metrics['F1'], metrics['Auc'])

                if print_trend:
                    early_stopping.draw_trend(train_loss_list, validation_loss_list, save_path=None)

                if save_loss:
                    csv_file_path = os.path.join(loss_trend_dir, setting + '-loss_data.csv')
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([setting, "total_train_epoch:", epoch, 'total_train_time', total_train_time])
                        writer.writerow([setting, 'train_loss_list'])
                        writer.writerow([setting, 'validation_loss_list'])
                        writer.writerow(train_loss_list)
                        writer.writerow(validation_loss_list)

                with open(total_perform_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    result = [relation_setting, GNN_model, r, epoch, metrics['Acc'], metrics['Pre'],
                              metrics['Rec'], metrics['F1'], metrics['Auc']]
                    writer.writerow(result)

                file_perform_per_round = os.path.join(perform_file_dir, setting + '-perform_data.npy')
                np.save(file_perform_per_round, metrics)

