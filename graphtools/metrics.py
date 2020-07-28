import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd #this will be used in the fscore function

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
        return pd.DataFrame(np.zeros(numerator.shape)) #all options shall return the same datatype


def compute_scores(y_test, y_pred, y_score,choice,
                   metrics=['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']):
    """

    @type metrics: list
    @param y_test: actual test labels
    @param y_pred: predicted test labels
    @param y_score: the scores for comparison between test and predicted labels
    @return:
    """
    scores = [balanced_accuracy_score(y_test, y_pred),
              accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')
              ]
    if "roc_auc_ovr_weighted" in metrics:
        if choice=='qcut':
            scores.append(roc_auc_score(y_test, y_score, average='weighted', multi_class='ovr'))
        else:
            scores.append(roc_auc_score(y_test, y_score, average='weighted'))
    return {k: v for k, v in zip(metrics, scores)}
