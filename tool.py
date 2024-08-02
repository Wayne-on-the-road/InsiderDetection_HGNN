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

def report_metrics(classifier, train_x, train_y, test_x, test_y):

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

