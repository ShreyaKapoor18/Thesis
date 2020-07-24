import datetime
import json
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from paramopt import get_distributions
from sklearn.model_selection import train_test_split
from metrics import fscore, compute_scores
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import PredefinedSplit
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
        # y.reset_index(drop=True, inplace=True)
        print(sum(y == 2), 'is the number of subjects which have been removed')
        y = y[y != 2]
        y = y // 3  # 0 and 1 classes get mapped to 0 and 3,4 get mapped to 1
        print(len(y), 'New number of subjects in our dataset')
        # X = whole[y.index, index[0]] don't know why this type of slicing is not working
        X = [X.loc[i, index] for i in list(y.index)]
    return np.array(X), np.array(y)

def split_vals(target_label, choice, train_test, train_bins):
    assert target_label.isna().any() == False
    if train_test == 'train':
        if choice == 'qcut':
            # choice to cut into three quartiles
            y, bins = pd.qcut(target_label, 3, labels=False, retbins=True)
            # return the values of these also so that the same ones can be used for the outof bag estimates
            return y,bins
        if choice == 'median':
            # choice to threshold around the median
            y = target_label >= target_label.median()
            bins = pd.qcut(target_label, 2, labels=False, retbins=True)[1]
            return y, bins
        if choice == 'throw median':
            y, bins = pd.qcut(target_label, 5, labels=False, retbins=True)
            # y.reset_index(drop=True, inplace=True)
            print(sum(y == 2), 'is the number of subjects which have been removed')
            y = y[y != 2]
            y = y // 3  # 0 and 1 classes get mapped to 0 and 3,4 get mapped to 1
            print(len(y), 'New number of subjects in our dataset')
            return y, bins
    if train_test == 'test':
        print('thresholds from the training data', train_bins)

        if choice == 'throw median':

            y = pd.cut(target_label, train_bins, labels=False, include_lowest=True)
            y = y // 3
            print('bins extreme',train_bins[0], train_bins[-1])
            return y

        y = pd.cut(target_label, train_bins, labels=False, include_lowest=True)
        y[y < train_bins[0]] = 0
        if choice == 'qcut':
            y[y > train_bins[-1]] = 2
            #print(y.isna().any())
            return y
        if choice == "median":
            y[y > train_bins[-1]] = 1
            #print(y.isna().any())
            return y

# %%
def dict_classifier(classifier, whole, metrics, target_col, edge):
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
    # note we are running for one label at a time!  # different labels
    for choice in ['throw median', 'qcut', 'median']:
        metric_score[choice] = {}
        best_params[choice] = {}
        clf, distributions = get_distributions(classifier)
        print(f'Executing {clf}')
        for per in [5, 10, 50, 100]:
            print('percentage', per)
            X_train_c, X_test, y_train_c, y_test = train_test_split(whole, target_col, test_size=0.1, random_state=55)
            assert len(X_test) == len(y_test)
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(X_train_c, y_train_c)
            print('Splits', skf)
            metric_score[choice][per] = {}
            best_params[choice][per] = {}
            test_scores = []
            rcv_scores =[]
            for train_index, val_index in skf.split(X_train_c, y_train_c):
                print("TRAIN:", len(train_index),train_index, "VALIDATION:", len(val_index), val_index)
                X_train, X_val = X_train_c.iloc[train_index,:], X_train_c.iloc[val_index,:]
                y_train, y_val = y_train_c.iloc[train_index], y_train_c.iloc[val_index]
                y_train, bins = split_vals(y_train, choice, 'train', None)
                # need to split training into training and validation set
                print('Y train shape after splitting values', y_train.shape)
                print('Y value shape', y_val.shape)
                bins[0] = 0
                bins[-1] = 100
                y_val = split_vals(y_val, choice, 'test', bins) #the bins depend on the training data
                print('y test before splitting', y_test)
                y_test = split_vals(y_test, choice, 'test', bins)

                print('y_test', y_test, 'y train', y_train, 'y val', y_val)
                if choice == 'throw median':
                    X_train = X_train.loc[y_train.index, :] # wouldn't need it now
                    X_val = X_val.loc[y_val.index, :] # before feature selection

                assert X_train.index.all() == y_train.index.all()
                stacked = pd.concat([X_train, y_train], axis=1)

                cols = []
                cols.extend(range(X_train.shape[1]))
                cols.append(target_col.name)
                stacked.columns = cols
                if edge == 'fscore':
                    arr = fscore(stacked, class_col=target_col.name)[:-1]
                    #fscore is different for the multiclass and binary case; has been incorporated above
                if edge == 'pearson':
                    arr = stacked.corr().iloc[:,-1]
                arr.fillna(0, inplace=True)
                arr = np.array(arr)
                #print(arr)
                val = np.nanpercentile(arr, 100 - per)
                #print('percentile value or threshold', val)
                index = np.where(arr >= val)
                X_train = X_train.iloc[:, index[0]]
                X_val = X_val.iloc[:, index[0]]
                assert len(X_train) == len(y_train)
                assert len(X_val) == len(y_val)

                y_train_comb = y_train.append(y_val)
                print('y train c index', y_train_c.shape)
                y_train_comb.sort_index(inplace=True)
                X_train_comb = pd.concat([X_train, X_val], axis=0)
                X_train_comb.sort_index(inplace=True, axis=0)
                split_index = [-1 if x in X_train.index else 0 for x in X_train_comb.index]

                # Use the list to create PredefinedSplit
                pds = PredefinedSplit(test_fold=split_index)

                rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                         refit='accuracy', cv=pds)
                print(X_train_comb.index, y_train_comb.index)
                print('X_train', 'X_val', 'X_combined', X_train.shape, X_val.shape, X_train_comb.shape)
                print('y_train', 'y_val', 'y_combined', y_train.shape, y_val.shape, y_train_comb.shape)
                print('reshaped y_train', np.array(y_train_comb).reshape(-1,1).shape)
                assert X_train.shape[1] == X_val.shape[1]
                search = rcv.fit(np.array(X_train_comb), np.array(y_train_comb))
                rcv_scores.append(search.cv_results_)
                y_pred = rcv.predict(X_test.iloc[:, index[0]])
                y_score =rcv.predict_proba(X_test.iloc[:, index[0]])
                if len(y_score[0]) == 2:
                    test_scores.append(compute_scores(y_test, y_pred, [x[1] for x in y_score], choice, metrics))
                else:
                    test_scores.append(compute_scores(y_test, y_pred, y_score, choice, metrics))
                best_params[choice][per] = search.best_params_ #need to see if this is in cv

            for metric in metrics:
                # validation set
                metric_score[choice][per][metric] = {}
                metric_score[choice][per][metric]['validation'] = round(np.mean([scores[f'mean_test_{metric}']for scores in rcv_scores]), 3)
                metric_score[choice][per][metric]['oob'] = np.mean([score[metric] for score in test_scores])
                print(f'validation {metric}', round(np.mean([scores[f'mean_test_{metric}']for scores in rcv_scores]), 3))
                print(f'oob error {metrc}',np.mean([score[metric] for score in test_scores]))
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
        legends.extend([choice+'_validation_score', choice+'_oob_score'])
    for k in range(len(top_per)):
        for j in range(len(metrics)):
            for choice, color in zip(['qcut', 'median', 'throw median'], ['orange', 'green', 'pink']):
                validation = []
                oob = []
                for clf in combined.keys():
                    validation.append(combined[clf][choice][top_per[k]][metrics[j]]['validation'])
                    oob.append(combined[clf][choice][top_per[k]][metrics[j]]['oob'])

                ax[k][j].plot(list(combined.keys()), validation, marker='+', label=choice + '_validation_score',
                              color=color, linestyle='dashed',markersize=12)
                ax[k][j].plot(list(combined.keys()), oob, marker='*', label=choice + '_oob_score', color=color
                              ,markersize=12)
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
        d1 = dict_classifier(clf, whole, metrics, target_col, edge)
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
    visualise_performance(combined, metrics, ['5', '10', '50', '100'], target)
