import datetime
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import describe
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from metrics import fscore, compute_scores
from metrics import stratify_sampling, plot_grid_search
from paramopt import get_distributions

# %%

def feature_selection(X_train, X_val, y_train, y_val, per, target_col, edge, ):
    scalar2 = StandardScaler()
    print('feature selection')
    # print('Initial X_train, y_train, X_val, y_val',  X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    X_train = pd.DataFrame(scalar2.fit_transform(X_train), index=X_train.index)
    X_train = pd.DataFrame(scalar2.fit_transform(X_train), index=X_train.index)
    X_val = pd.DataFrame(scalar2.transform(X_val), index=X_val.index)

    stacked = pd.concat([X_train, y_train], axis=1)
    cols = []
    cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
    cols.append(target_col.name)
    stacked.columns = cols
    if edge == 'fscore':
        arr_inner = fscore(stacked, class_col=target_col.name)[:-1]
        # fscore is different for the multiclass and binary case; has been incorporated above
    if edge == 'pearson':
        arr_inner = stacked.corr().iloc[:-1, -1]
    arr_inner.fillna(0, inplace=True)
    arr_inner = np.array(arr_inner)
    val = np.nanpercentile(arr_inner, 100 - per)
    index = np.where(arr_inner >= val)

    X_train = X_train.iloc[:, index[0]]
    X_val = X_val.iloc[:, index[0]]
    # print('X_train_c', X_train_c.shape, 'y_train_c', y_train_c.shape)
    # print('X_test', X_test.shape, 'y_test', y_test.shape)
    assert list(X_train.index) == list(y_train.index)
    #assert list(X_val.index) == list(y_val.index)
    print('Final X_train,X_test, y_train, y_test', X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    return X_train, X_val


def make_predefined_split(X_train, X_val, y_train, y_val):
    y_train_comb = y_train.append(y_val)
    y_train_comb.sort_index(inplace=True)
    X_train_comb = pd.concat([X_train, X_val], axis=0)
    X_train_comb.sort_index(inplace=True, axis=0)
    assert list(X_train_comb.index) == list(y_train_comb.index)
    assert set(X_train_comb.index) == set(X_val.index).union(set(X_train.index))
    split_index = [-1 if x in X_train.index else 0 for x in X_train_comb.index]
    return X_train_comb, y_train_comb, split_index


# %%
def dict_classifier(classifier, whole, metrics, target_col, edge, percent):
    """

    :param whole: the matrix containing the edge information for all subjects
    :param classifier: the name of the classifier we want to test
    :param metrics: the name of the metrics we want to calculate

    :return: metric_scores: the values to be calculated using permitted keywords
    """
    refit_metric = 'balanced_accuracy'
    print('The refit metric we are using', refit_metric)
    print('=' * 100)
    print('Classifier', classifier)
    metric_score = {}
    best_params = {}

    return {'Metrics': metric_score, 'Parameters': best_params}


# %%
def make_csv(dict_score, filename):
    """

    @param dict_score: the dictionary of scores
    @param filename: the name of the file where the csv is stored
    """
    cv1 = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score['Metrics'].items()
    },
        axis=1)
    cv2 = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score['Parameters'].items()
    },
        axis=1)
    cv1.to_csv(filename)
    cv2.to_csv(filename.split('.csv')[0] + '_bestparams.csv')


# %%
def visualise_performance(combined, metrics, top_per, target, choices, classifiers):
    """

    @param combined: the dictionary that contains the scores for all possibilities
    @param big5: the big5 personality traits
    @param metrics: the metrics we want to calculate for the data
    @param top_per: the top percentile of the features we want to use
    """
    # for each label we will visualise the performance of different classifiers
    fig, ax = plt.subplots(len(top_per), len(metrics), figsize=(12, 12))
    legends = []
    for choice in choices:
        legends.extend([choice + 'test'])
        legends.extend([choice + 'train'])
    for k in range(len(top_per)):
        for j in range(len(metrics)):
            for color, choice in zip(['orange', 'green', 'pink'][:len(choices)], choices):
                test_score = []
                train_score = []
                for clf in classifiers:
                    # print("dictionary", combined[clf][choice])
                    test_score.append(combined[clf][top_per[k]][choice][metrics[j]]['test'])
                    train_score.append(combined[clf][top_per[k]][choice][metrics[j]]['train'])

                ax[k][j].plot(list(combined.keys()), test_score, marker='+', label=choice + 'test',
                              color=color, linestyle='dashed', markersize=12)
            if top_per[k] == 0:
                ax[k][j].set_title(f'100% features')
            else:
                ax[k][j].set_title(f'Top {top_per[k]}% features')
            ax[k][j].set_xlabel('Classifier')
            ax[k][j].set_ylabel(metrics[j])
            ax[k][j].grid(which='minor', alpha=0.2)
            ax[k][j].grid(which='major', alpha=0.5)
    plt.legend(legends)
    fig.suptitle(target)
    plt.tight_layout()
    plt.savefig(f'outputs/figures/classification_{target}')
    plt.show()


# %%
def run_classification(whole, metrics, target, target_col, edge):
    """
    @param label: the target variable since we want to run in a pipeline
    @param whole: the oriignal data with combined features from all subjects
    @param metrics: the metrics we want to calculate for the predictions
    @param labels: the original label in the target dataframe
    """
    combined = {}
    best_params_combined = {}
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    for folder in ['figures', 'dicts', 'csvs']:
        if not os.path.exists(f'outputs/{folder}'):
            os.mkdir(f'outputs/{folder}')
    classifiers  = ['SVC', 'RF', 'MLP']
    for clf in classifiers:  # other ones are taking too long
        start = time.time()
        d1 = dict_classifier(clf, whole, metrics, target_col, edge, [5, 10, 50, 100])
        end = time.time()
        print(f'Time taken for {clf}: {datetime.timedelta(seconds=end - start)}')
        make_csv(d1, f'outputs/csvs/{target}_{clf}_results_cv.csv')
        with open(f'outputs/dicts/{target}_{clf}_results_cv.json', 'w') as fp:
            json.dump(d1, fp, indent=4)
        combined[clf] = d1['Metrics']
        best_params_combined[clf] = d1['Parameters']

    with open(f'outputs/dicts/{target}_combined_dict.json', 'w') as f:
        # write the combined dictionary to the file so that this can be read later on
        json.dump(combined, f, indent=4)
    with open(f'outputs/dicts/{target}_combined_params.json', 'w') as f:  #
        json.dump(best_params_combined, f, indent=4)
    try:
        visualise_performance(combined, metrics, [5, 10, 50, 100], target,
                              ['test throw median', 'keep median'], classifiers)
    except KeyError:
        print("There was a key value error in the first case")
        try:
            visualise_performance(combined, metrics, ['5', '10', '50', '100'], target,
                                  ['test throw median', 'keep median'] , classifiers)
        except KeyError:
            print("Again couldn't visualize")
