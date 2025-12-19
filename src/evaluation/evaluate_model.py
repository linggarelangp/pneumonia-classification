from sklearn.metrics import *

def evaluate_model(
        y_true,
        y_pred,
    ):
    """
    Evaluate the performance of a classification model.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'auc': auc,
    }