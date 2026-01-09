import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def classification_metrics(y_true, y_pred, y_prob):
    """
    Standard predictive accuracy metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob)
    }


def stability_physics_metrics(df_original, y_pred):
    """
    Power-system-aware stability metrics.
    """

    vsi_error = np.mean(np.abs(df_original["VSI"] - (1 - y_pred)))
    q_margin_dev = np.mean(np.abs(df_original["Q_margin"]))
    freq_rms_error = np.mean(np.abs(df_original["FreqDev_RMS"]))

    return {
        "vsi_error": vsi_error,
        "q_margin_deviation": q_margin_dev,
        "frequency_rms_error": freq_rms_error
    }
