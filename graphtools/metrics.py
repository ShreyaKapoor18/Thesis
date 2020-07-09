import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# %%
def fscore(data, class_col='class'):
    """ Compute the F-score for all columns in a DataFrame
    @param data: the dataframe of containing the target values
    @param class_col: the column by which we want to sort the dataframe
    @return:
    """
    grouped = data.groupby(by=class_col)
    means = data.mean()
    g_means = grouped.mean()
    g_vars = grouped.var()

    numerator = np.sum((g_means - means) ** 2, axis=0)
    denominator = np.sum(g_vars, axis=0)
    if sum(denominator) != 0:
        return round(numerator / denominator, 3)
    else:
        return np.zeros(numerator.shape)


def compute_scores(y_test, y_pred, y_score):
    """

    @param y_test: actual test labels
    @param y_pred: predicted test labels
    @param y_score: the scores for comparison between test and predicted labels
    @return:
    """
    scores = [balanced_accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_score),
              accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)]
    return scores
