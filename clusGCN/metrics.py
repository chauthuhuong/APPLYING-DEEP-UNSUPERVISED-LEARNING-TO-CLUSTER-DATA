from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return nmi, ari, acc
