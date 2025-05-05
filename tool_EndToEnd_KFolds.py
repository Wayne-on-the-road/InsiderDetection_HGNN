import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,auc,roc_curve



def draw_roc(true_y,pos_score,line_label):
    fpr_g, tpr_g, threshold_g = roc_curve(true_y, pos_score)
    roc_auc = auc(fpr_g, tpr_g)
    lw = 1.5
    plt.plot(fpr_g, tpr_g, color='red',
             lw=lw, label=f'S{line_label} (AUC = %0.4f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random', x=0.6, y=0.4)
    plt.legend(loc="lower right")
    plt.show()

def classifier_report(classifier, train_x, train_y, test_x, test_y):

    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)
    pos_score = classifier.predict_proba(test_x)[:, 1]
    true_y = test_y

    accuracy = accuracy_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y)

    fpr, tpr, threshold = roc_curve(true_y, pos_score)
    auc_score = auc(fpr, tpr)
    metrics = {
        'Acc': accuracy,
        'Pre': precision,
        'Rec': recall,
        'F1': f1,
        'Auc': auc_score,
        'pos_score': pos_score,
        'fpr': fpr,
        'tpr': tpr,
    }

    return metrics #, precision, recall, f1, pos_score, auc,



def report_metrics(ground_truth, pred_class, pred_proba, mode='train'):
    """
    Reports the metrics for classification in the required format.

    Args:
    - ground_truth (np.array): True labels.
    - pred_class (np.array): Predicted class labels.
    - pred_proba (np.array): Predicted probabilities for the positive class.
    - mode (str): 'train' to report only essential metrics, 'test' to report all metrics.

    Returns:
    - metrics_dict (dict): A dictionary containing the requested metrics.
    """
    # Define the metric functions for each metric
    metric_functions = {
        'Acc': lambda: accuracy_score(ground_truth, pred_class),
        'Pre': lambda: precision_score(ground_truth, pred_class),
        'Rec': lambda: recall_score(ground_truth, pred_class),
        'F1': lambda: f1_score(ground_truth, pred_class),
        'Auc': lambda: roc_auc_score(ground_truth, pred_proba),
        'pos_score': lambda: pred_proba,
        'fpr': lambda: roc_curve(ground_truth, pred_proba)[0],
        'tpr': lambda: roc_curve(ground_truth, pred_proba)[1]
    }

    # Choose which metrics to compute based on the mode
    if mode == 'train':
        selected_metrics = ['Acc', 'Pre', 'Rec', 'F1', 'Auc']
    elif mode == 'test':
        selected_metrics = ['Acc', 'Pre', 'Rec', 'F1', 'Auc', 'pos_score', 'fpr', 'tpr']
    else:
        raise ValueError("Mode should be 'train' or 'test'.")

    # Compute the selected metrics using the pre-defined functions
    metrics_dict = {metric: metric_functions[metric]() for metric in selected_metrics}

    return metrics_dict

