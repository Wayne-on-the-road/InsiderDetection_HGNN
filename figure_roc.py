import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Define the file path and classifier names manually
perform_file_dir = './result_dir/All_relation/GCN/perform_file'
Round = [22, 27, 12] # corresponding to the classifier's median performance round (15th place))
classifier_name_list = ["GNB", "SVM", "LR"]
line_weight= 1.5
font_size = 14
figure_path = r'd:\Users\Wayne\Desktop\PhD Research\2024_WISE\graph_and_table\roc_curves'

# Define color-blind friendly colors
color_blind_friendly_colors = [
    '#377eb8',  # Blue
    '#ff7f00',  # Orange
    '#4daf4a',  # Green
    '#f781bf',  # Pink
    '#a65628',  # Brown
    '#984ea3',  # Purple
    '#999999',  # Grey
    '#e41a1c',  # Red
    '#dede00'   # Yellow
]

# Set the color cycle to use the color-blind friendly colors
plt.rc('axes', prop_cycle=(cycler('color', color_blind_friendly_colors)))
for i, r in enumerate(Round):
    setting = f'All_relation-GCN-BS_2048-HC_16-{r}th'
    file_perform_per_round = os.path.join(perform_file_dir, setting + '-perform_data.npy')
    # Load the performance data
    perform_per_round = np.load(file_perform_per_round, allow_pickle=True).item()
    # Plot and save ROC curves for each classifier
    for j, classifier_name in enumerate(classifier_name_list):
        metrics_original = perform_per_round[classifier_name]['original']
        metrics_with_graph = perform_per_round[classifier_name]['with_graph']

        fpr_original = metrics_original['fpr']
        tpr_original = metrics_original['tpr']
        roc_auc_original = metrics_original['Auc']

        fpr_with_graph = metrics_with_graph['fpr']
        tpr_with_graph = metrics_with_graph['tpr']
        roc_auc_with_graph = metrics_with_graph['Auc']

        plt.figure()
        plt.plot(fpr_original, tpr_original, lw=line_weight, label=f'Xb (Auc = {roc_auc_original:.4f})',
                 color=color_blind_friendly_colors[i], linestyle='--')
        plt.plot(fpr_with_graph, tpr_with_graph, lw=line_weight, label=f'Xb+Xg (Auc = {roc_auc_with_graph:.4f})',
                 color=color_blind_friendly_colors[i+4])
        plt.plot([0, 1], [0, 1], color='grey', lw=line_weight, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=font_size)
        plt.ylabel('True Positive Rate', fontsize=font_size)
        plt.title(f'ROC comparison for {classifier_name}', fontsize=font_size)
        plt.legend(loc="lower right", fontsize=font_size)
        plt.grid(True)
        plt.tight_layout()


        # Save the figure
        save_path = os.path.join(figure_path, setting + f'{classifier_name}_ROC_Comparison.pdf')
        plt.savefig(save_path, dpi=600, format='pdf')
        plt.show()
        plt.close()