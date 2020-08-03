import datetime
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from metrics import fscore, compute_scores
from paramopt import get_distributions


# %%
def data_splitting(choice, index, X, y):
    """
     @param choice: the choice of how to split the target values
     @param index: feature indices
     @return: X,y the modified data and the target values
     """
    if choice == 'qcut':
        # choice to cut into three quartiles
        y = pd.qcut(y, 3, labels=False, retbins=True)[0]
        X = X.iloc[:, index]
    if choice == 'median':
        # choice to threshold around the median
        y = y >= y.median()
        X = X.iloc[:, index]
    if choice == 'throw median':
        y = pd.qcut(y, 5, labels=False, retbins=True)[0]

        print(sum(y == 2), 'is the number of subjects which have been removed')
        y = y[y != 2]
        y = y // 3  # 0 and 1 classes get mapped to 0 and 3,4 get mapped to 1
        print(len(y), 'New number of subjects in our dataset')
        # X = whole[y.index, index[0]] don't know why this type of slicing is not working
        X = pd.DataFrame([X.loc[i, index] for i in list(y.index)])
    return X, y


def feature_selection(X_train, X_val, y_train, y_val, per, target_col, edge, ):
    scalar2 = StandardScaler()
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
        arr_inner = stacked.corr().iloc[:, -1]
    arr_inner.fillna(0, inplace=True)
    arr_inner = np.array(arr_inner)
    val = np.nanpercentile(arr_inner, 100 - per)
    index = np.where(arr_inner >= val)

    X_train = X_train.iloc[:, index[0]]
    X_val = X_val.iloc[:, index[0]]
    # print('X_train_c', X_train_c.shape, 'y_train_c', y_train_c.shape)
    # print('X_test', X_test.shape, 'y_test', y_test.shape)
    assert list(X_train.index) == list(y_train.index)
    assert list(X_val.index) == list(y_val.index)
    return X_train, X_val


def make_predefined_split(X_train, X_val, y_train, y_val):
    y_train_comb = y_train.append(y_val)
    y_train_comb.sort_index(inplace=True)
    X_train_comb = pd.concat([X_train, X_val], axis=0)
    X_train_comb.sort_index(inplace=True, axis=0)
    split_index = [-1 if x in X_train.index else 0 for x in X_train_comb.index]
    return X_train_comb, y_train_comb, split_index


# %%
def dict_classifier(classifier, whole, metrics, target_col, edge, percent):
    """
    Nested Cross validation is needed, the hyperparmeters are obtained from the inner cross validation
    :param whole: the matrix containing the edge information for all subjects
    :param option: if we want the scores with or without cross validation
    :param classifier: the name of the classifier we want to test
    :param metrics: the name of the metrics we want to calculate
    :param labels: the big5 personality labels
    :param data: the file with contains the labels for all features of all subjects
    :param new_fscores: flattened array of f scores: num_subjects x num edges
    :return: metric_scores: the values to be calculated using permitted keywords
    """
    print('Classifier', classifier)
    metric_score = {}
    best_params = {}
    # note we are running for one label at a time!  # different labels
    for choice in ['throw median', 'qcut', 'median']:
        print(choice)
        metric_score[choice] = {}
        best_params[choice] = {}
        X, y = data_splitting(choice, range(whole.shape[1]), whole, target_col)
        for per in percent:
            print('percentage', per)
            outer_cv = StratifiedKFold(n_splits=5)
            inner_cv = StratifiedKFold(n_splits=10)
            metric_score[choice][per] = {}
            best_params[choice][per] = {}
            outer_cv_scores = []
            outer_cv_params = []
            print('the number of splits', outer_cv.get_n_splits())
            for train_index, test_index in outer_cv.split(X, y):
                inner_cv_scores = []
                inner_cv_params = []
                print('Outer cv loop')
                # print('X', X.shape)
                X_train_c, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train_c, y_test = y.iloc[train_index], y.iloc[test_index]
                print('Xtrain, Xtest, ytrain, ytest', X_train_c.shape, X_test.shape, y_train_c.shape, y_test.shape)

                for train_idx, val_idx in inner_cv.split(X_train_c, y_train_c):
                    X_train, X_val = X_train_c.iloc[train_idx, :], X_train_c.iloc[val_idx, :]
                    y_train, y_val = y_train_c.iloc[train_idx], y_train_c.iloc[val_idx]
                    print('X train, ytrain, Xval, yval', X_train.shape, y_train.shape, X_val.shape, y_val.shape)

                    X_train, X_val = feature_selection(X_train, X_val, y_train, y_val, per, target_col, edge)
                    X_train_comb, y_train_comb, split_index = make_predefined_split(X_train, X_val, y_train, y_val)
                    # print('split index', split_index, 'need to check if this is correct or not')
                    # Use the list to create PredefinedSplit
                    pds = PredefinedSplit(test_fold=split_index)
                    # print('number of splits', pds.get_n_splits(), pds)
                    clf, distributions = get_distributions(classifier, True, None)
                    rcv = RandomizedSearchCV(clf, distributions, random_state=45, scoring=metrics,
                                             refit='roc_auc_ovr_weighted', cv=pds, n_iter=1000,
                                             n_jobs=-1)  # this is already producing 5 folds so we need to do something different?
                    # both the training and the validation set shall be passed since we have passed the indices of the validation set
                    search = rcv.fit(X_train_comb, y_train_comb)
                    inner_cv_scores.append(search.best_score_)
                    inner_cv_params.append(search.best_estimator_)
                    # print('rcv scores length', len(search.cv_results_['mean_test_roc_auc_ovr_weighted']), search.cv_results_['mean_test_roc_auc_ovr_weighted'])
                    print('best internal cv params', search.best_params_)

                X_train_c, X_test = feature_selection(X_train_c, X_test, y_train_c, y_test, per, target_col, edge)
                assert len(inner_cv_scores) == len(inner_cv_params)
                # print('keys', inner_cv_scores[0].keys())
                clf1 = inner_cv_params[np.argmax([inner_cv_scores])]
                # we have this because of the number of trials for the parameter settings
                clf1.fit(X_train_c, y_train_c)
                y_pred = clf1.predict(X_test)
                y_score = clf1.predict_proba(X_test)

                if len(y_score[0]) == 2:  # number of classes in the y_score
                    outer_cv_scores.append(compute_scores(y_test, y_pred, [x[1] for x in y_score], choice, metrics))
                    # The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.S
                else:
                    outer_cv_scores.append(compute_scores(y_test, y_pred, y_score, choice, metrics))
                outer_cv_params.append(search.best_params_)

            print('outer cv scores completed', outer_cv_scores)
            print('outer cv balanced acc', [score['balanced_accuracy'] for score in outer_cv_scores])
            loc = np.argmax([score['roc_auc_ovr_weighted'] for score in outer_cv_scores])
            best_params[choice][per] = outer_cv_params[loc]  # need to see if this is in cv

            for metric in metrics:
                # validation set
                metric_score[choice][per][metric] = {}
                # but the best score into the outer cv part instead of the average
                metric_score[choice][per][metric] = round(outer_cv_scores[loc][metric], 3)
                print(f'{metric}', np.mean([score[metric] for score in outer_cv_scores]))
                # out of bag error
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
def visualise_performance(combined, metrics, top_per, target):
    """

    @param combined: the dictionary that contains the scores for all possibilities
    @param big5: the big5 personality traits
    @param metrics: the metrics we want to calculate for the data
    @param top_per: the top percentile of the features we want to use
    """
    # for each label we will visualise the performance of different classifiers
    fig, ax = plt.subplots(len(top_per), len(metrics), figsize=(20, 20))
    legends = []
    for choice in ['qcut', 'median', 'throw median']:
        legends.extend([choice + '_validation_score', choice + '_oob_score'])
    for k in range(len(top_per)):
        for j in range(len(metrics)):
            for choice, color in zip(['qcut', 'median', 'throw median'], ['orange', 'green', 'pink']):
                validation = []
                for clf in combined.keys():
                    print("dictionary", combined[clf][choice])
                    validation.append(combined[clf][choice][top_per[k]][metrics[j]])

                ax[k][j].plot(list(combined.keys()), validation, marker='+', label=choice,
                              color=color, linestyle='dashed', markersize=12)
            if top_per[k] == 0:
                ax[k][j].set_title(f'100% features')
            else:
                ax[k][j].set_title(f'Top {top_per[k]}% features')
            ax[k][j].set_xlabel('Classifier')
            ax[k][j].set_ylabel(metrics[j])
            ax[k][j].grid()

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

    for clf in ['SVC', 'RF', 'GB', 'MLP'][:2]:  # other ones are taking too long
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
        visualise_performance(combined, metrics, [5, 10, 50, 100], target)
    except KeyError:
        print("There was a key value error in the first case")
        try:
            visualise_performance(combined, metrics, ['5', '10', '50', '100'], target)
        except KeyError:
            print("Again couldn't visualize")
