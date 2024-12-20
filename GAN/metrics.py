import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def calculate_metrics(y_true, y_pred):
    """
    Compute NMI, ARI, and Accuracy.
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = np.sum(y_true == y_pred) / len(y_true)  # Simplified accuracy
    return {"NMI": nmi, "ARI": ari, "Accuracy": acc}
