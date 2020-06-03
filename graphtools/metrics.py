import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
#%%
def fscore(data, class_col='class'):
    """ Compute the F-score for all columns in a DataFrame
    """
    grouped = data.groupby(by=class_col)
    means = data.mean()
    g_means = grouped.mean()
    g_vars = grouped.var()

    numerator = np.sum((g_means - means) ** 2, axis=0)
    denominator = np.sum(g_vars, axis=0)
    if sum(denominator)!= 0:
        return round(numerator/denominator, 3)
    else:
        return np.zeros(numerator.shape)

def compute_scores(y_test, y_pred, y_score):
    scores = [balanced_accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_score),
              accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)]
    return scores