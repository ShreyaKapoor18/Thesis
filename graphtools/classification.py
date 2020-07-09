import datetime
import json
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from paramopt import get_distributions


# %%
def data_splitting(choice, i, index, data, whole, labels, *args, **kwargs):
    """

    @param choice: the choice of how to split the target values
    @param i:location of the personality trait
    @param index: feature indices
    @param data: dataframe from which we get access to the labels
    @param whole: training data with three types of features
    @param labels: original names of the features in the dataset
    @param args:
    @param kwargs:
    @return: X,y the modified data and the target values
    """
    if choice == 'qcut':
        # choice to cut into three quartiles
        y = pd.qcut(data[labels[i]], 3, labels=False, retbins=True)[0]
        X = whole.iloc[:, index]
    if choice == 'median':
        # choice to threshold around the median
        y = data[labels[i]] >= data[labels[i]].median()
        X = whole.iloc[:, index]
    if choice == 'throw median':
        y = pd.qcut(data[labels[i]], 5, labels=False, retbins=True)[0]
        # y.reset_index(drop=True, inplace=True)
        print(sum(y == 2), 'is the number of subjects which have been removed')
        y = y[y != 2]
        y = y // 3  # 0 and 1 classes get mapped to 0 and 3,4 get mapped to 1
        print(len(y), 'New number of subjects in our dataset')
        # X = whole[y.index, index[0]] don't know why this type of slicing is not working
        X = [whole.loc[i, index] for i in list(y.index)]
        # X = whole.iloc[y.index, index[0]]

    return np.array(X), np.array(y)


# %%
def dict_classifier(classifier, big5, new_fscores, data, whole, metrics, labels):
    """
    :param whole: the matrix containing the edge information for all subjects
    :param option: if we want the scores with or without cross validation
    :param classifier: the name of the classifier we want to test
    :param metrics: the name of the metrics we want to calculate
    :param labels: the big5 personality labels
    :param data: the file with contains the labels for all features of all subjects
    :param new_fscores: flattened array of f scores: num_subjects x num edges
    :return: metric_scores: the values to be calculated using permitted keywords
    """
    metric_score = {}
    best_params = {}
    for i in range(5):  # different labels
        # print(labels[i], ':', big5[i])
        metric_score[big5[i]] = {}
        best_params[big5[i]] = {}

        for per in [5, 10, 50, 100]:
            print('percentage', per)
            metric_score[big5[i]][per] = {}
            best_params[big5[i]][per] = {}
            val = np.percentile(new_fscores[i], 100 - per)
            index = np.where(new_fscores[i] >= val)

            # print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))
            # Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
            for choice in ['qcut', 'median', 'throw median']:
                metric_score[big5[i]][per][choice] = {}
                X, y = data_splitting(choice, i, index[0], data, whole, labels)
                clf, distributions = get_distributions(classifier)
                print(f'Executing {clf}')
                # roc doesn't support multiclass
                rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                         refit='roc_auc_ovr_weighted', cv=5)
                # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                search = rcv.fit(X, y)
                scores = search.cv_results_
                best_params[big5[i]][per][choice] = search.best_params_
                for metric in metrics:
                    metric_score[big5[i]][per][choice][metric] = round(np.mean(scores[f'mean_test_{metric}']), 3)

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
def visualise_performance(combined, big5, metrics, top_per):
    """

    @param combined: the dictionary that contains the scores for all possibilities
    @param big5: the big5 personality traits
    @param metrics: the metrics we want to calculate for the data
    @param top_per: the top percentile of the features we want to use
    """
    # for each label we will visualise the performance of different classifiers
    for i in range(len(big5)):
        fig, ax = plt.subplots(len(top_per), len(metrics), figsize=(25, 20))
        for k in range(len(top_per)):
            for j in range(len(metrics)):

                for choice in ['qcut', 'median', 'throw median']:
                    test = []

                    for clf in combined.keys():
                        test.append(combined[clf][big5[i]][top_per[k]][choice][metrics[j]])
                        # print(clf, big5[i], top_per[k], metrics[j])
                        # print(combined[clf][big5[i]][top_per[k]][metrics[j]])
                    ax[k][j].scatter(combined.keys(), test)
                    ax[k][j].plot(list(combined.keys()), test, marker='+', label=choice + '_test_score')
                    # print('xx', len(l))
                ax[k][j].legend(loc='lower right')
                # ax[k][j].set_xticks(list(combined.keys()))
                if top_per[k] == 0:
                    ax[k][j].set_title(f'100% features')
                else:
                    ax[k][j].set_title(f'Top {top_per[k]}% features')
                ax[k][j].set_xlabel('Classifier')
                ax[k][j].set_ylabel(metrics[j])
                ax[k][j].grid()
        fig.suptitle(big5[i])
        plt.tight_layout()
        plt.savefig(f'outputs/figures/classification_{big5[i]}')
        # plt.show()


# %%
def run_classification(whole, metrics, big5, data, new_fscores, labels):
    """

    @param whole: the oriignal data with combined features from all subjects
    @param metrics: the metrics we want to calculate for the predictions
    @param big5: the big5 personality traits labels
    @param data: the labels for the data
    @param new_fscores: fscores for all features rescaled
    @param labels: the original labels in the target dataframe
    """
    combined = {}
    best_params_combined = {}
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
        for folder in ['figures', 'dicts', 'csvs']:
            if not os.path.exists(f'outputs/{folder}'):
                os.mkdir(f'outputs/{folder}')

    for clf in ['SVC', 'RF', 'GB', 'MLP']:  # other ones are taking too long
        start = time.time()
        d1 = dict_classifier(clf, big5, new_fscores, data, whole, metrics, labels)
        end = time.time()
        print(f'Time taken for {clf}: {datetime.timedelta(seconds=end - start)}')
        make_csv(d1, f'outputs/csvs/{clf}_results_cv.csv')

        with open(f'outputs/dicts/{clf}_results_cv.json', 'w') as fp:
            json.dump(d1, fp, indent=4)

        combined[clf] = d1['Metrics']
        best_params_combined[clf] = d1['Parameters']

    with open('outputs/dicts/combined_dict.json', 'w') as f:
        # write the combined dictionary to the file so that this can be read later on
        json.dump(combined, f, indent=4)
    with open('outputs/dicts/combined_params.json', 'w') as f:  #
        json.dump(best_params_combined, f, indent=4)
    visualise_performance(combined, big5, metrics, [5, 10, 50, 100])
