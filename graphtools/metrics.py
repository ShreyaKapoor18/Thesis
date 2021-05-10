import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd #this will be used in the fscore function
import matplotlib.pyplot as plt

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


def compute_scores(y_test, y_pred, y_score,
                   metrics=['balanced_accuracy', 'accuracy', 'f1_weighted', 'r', 'auc']):
    """

    @type metrics: list
    @param y_test: actual test labels
    @param y_pred: predicted test labels
    @param y_score: the scores for comparison between test and predicted labels
    @return:
    """
    #assert list(y_pred.index) == list(y_test.index)
    assert len(y_test) == len(y_pred)

    #fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=pos_label)
    scores = [balanced_accuracy_score(y_test, y_pred),
              accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
              roc_auc_score(y_test, y_score)]

    return {k: v for k, v in zip(metrics, scores)}


def stratify_sampling(x, n_samples, stratify):
    """Perform stratify sampling of a tensor.
    From  YannDubs/statify_sampling.py
    parameters
    ----------
    x: np.ndarray or torch.Tensor
        Array to sample from. Sampels from first dimension.

    n_samples: int
        Number of samples to sample

    stratify: tuple of int
        Size of each subgroup. Note that the sum of all the sizes
        need to be equal to `x.shape[']`.
    """
    n_total = x.shape[0]
    assert sum(stratify) == n_total

    n_strat_samples = [int(i * n_samples / n_total) for i in stratify]
    cum_n_samples = np.cumsum([0] + list(stratify))
    sampled_idcs = []
    for i, n_strat_sample in enumerate(n_strat_samples):
        sampled_idcs.append(np.random.choice(range(cum_n_samples[i], cum_n_samples[i + 1]),
                                             replace=False,
                                             size=n_strat_sample))

    # might not be correct number of samples due to rounding
    n_current_samples = sum(n_strat_samples)
    if n_current_samples < n_samples:
        delta_n_samples = n_samples - n_current_samples
        # might actually resample same as before, but it's only for a few
        sampled_idcs.append(np.random.choice(range(n_total), replace=False, size=delta_n_samples))

    samples = x.iloc[np.concatenate(sampled_idcs)]

    return samples

def plot_grid_search(cv_results, refit_metric):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results[f'mean_test_{refit_metric}']


    scores_sd = cv_results[f'std_test_{refit_metric}']

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(np.arange(len(scores_mean)), scores_mean, '-o')

    ax.set_title("Randomized Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel('Iteration Number', fontsize=16)
    ax.set_ylabel(f'{refit_metric} Score', fontsize=16)
    ax.grid('on')
    plt.show()


def diag_flattened_indices(a):
    '''
    Flattened array contains the upper diagonal matrix and we need to get the indices
    in the flattened array which correspond to the diagonal elements
    This returns only the diagonal elements
    @param n: the shape of the array
    @return:
    '''
    indices = []
    i = a
    n = 0
    while n < (a * (a + 1)) / 2:
        indices.append(n)
        n += i
        i -= 1
    return indices

def find_indices(lin, shape=84):
    d1 = {}
    mat = np.triu_indices(shape)
    for idx in lin:
        for i in range(len(mat[0])):
            if i == idx:
                d1[idx] = (mat[0][i], mat[1][i])
    return d1