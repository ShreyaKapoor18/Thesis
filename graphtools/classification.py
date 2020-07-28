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
from sklearn.preprocessing import StandardScaler
from PIL import Image
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
    return X,y

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
    myImage = Image.open("support/algo_nested_cv.png")
    myImage.show()
    metric_score = {}
    best_params = {}
    # note we are running for one label at a time!  # different labels
    for choice in ['throw median', 'qcut', 'median']:
        metric_score[choice] = {}
        best_params[choice] = {}
        clf, distributions = get_distributions(classifier)
        X,y = data_splitting(choice, range(whole.shape[1]), whole, target_col)
        print(f'Executing {clf}')
        for per in percent:
            print('percentage', per)
            outer_cv = StratifiedKFold(n_splits=10)
            inner_cv = StratifiedKFold(n_splits=5)

            metric_score[choice][per] = {}
            best_params[choice][per] = {}
            outer_cv_scores = []
            outer_cv_params = []
            for train_index, test_index in outer_cv.split(X,y):
                print('X', X.shape)
                X_train_c, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                y_train_c, y_test = y.iloc[train_index], y.iloc[test_index]
                print('X_train_c', X_train_c.shape, 'y_train_c', y_train_c.shape)
                print('X_test', X_test.shape, 'y_train_c', y_test.shape)
                inner_cv_scores = []
                inner_cv_params = []
                for train_idx, val_idx in inner_cv.split(X_train_c, y_train_c):
                    X_train, X_val = X_train_c.iloc[train_idx,:], X_train_c.iloc[val_idx,:]
                    y_train, y_val = y_train_c.iloc[train_idx], y_train_c.iloc[val_idx]
                    if choice == 'throw median': #only in this case are we throwing away some subjects
                        X_train = X_train.loc[y_train.index, :] # wouldn't need it now
                        X_val = X_val.loc[y_val.index, :] # before feature selection


                    scalar = StandardScaler()
                    X_train = pd.DataFrame(scalar.fit_transform(X_train), index= X_train.index)
                    X_val = pd.DataFrame(scalar.transform(X_val), index = X_val.index)
                    assert X_train.index.all() == y_train.index.all()

                    stacked = pd.concat([X_train, y_train], axis=1)
                    cols = []
                    cols.extend(range(X_train.shape[1])) # the values zero to the number of columns
                    cols.append(target_col.name)
                    stacked.columns = cols
                    if edge == 'fscore':
                        arr = fscore(stacked, class_col=target_col.name)[:-1]
                        #fscore is different for the multiclass and binary case; has been incorporated above
                    if edge == 'pearson':
                        arr = stacked.corr().iloc[:,-1]

                    arr.fillna(0, inplace=True)
                    arr = np.array(arr)
                    val = np.nanpercentile(arr, 100 - per)
                    index = np.where(arr >= val)
                    X_train = X_train.iloc[:, index[0]]
                    X_val = X_val.iloc[:, index[0]]

                    #feature selection based on fscore or pearson correlation coefficient
                    assert len(X_train) == len(y_train)
                    assert len(X_val) == len(y_val)

                    y_train_comb = y_train.append(y_val)
                    y_train_comb.sort_index(inplace=True)
                    X_train_comb = pd.concat([X_train, X_val], axis=0)
                    X_train_comb.sort_index(inplace=True, axis=0)
                    split_index = [-1 if x in X_train.index else 0 for x in X_train_comb.index]
                    print('split index',split_index, 'need to check if this is correct or not')
                    # Use the list to create PredefinedSplit
                    pds = PredefinedSplit(test_fold=split_index)

                    rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                             refit='balanced_accuracy', cv=pds) #this is already producing 5 folds so we need to do something different?
                    # both the training and the validation set shall be passed since we have passed the indices of the validation set

                    assert X_train.shape[1] == X_val.shape[1]
                    search = rcv.fit(np.array(X_train_comb), np.array(y_train_comb))
                    inner_cv_params.append(search.best_params_)
                    inner_cv_scores.append(np.mean(search.cv_results_['mean_test_balanced_accuracy']))
                    # select the parameters which correspond to the best results, fscores also depend on the split which is the best one, need to store that
                assert len(inner_cv_params) == len(inner_cv_scores)
                print('scores and parameters',inner_cv_scores, inner_cv_params)
                bestp_index = np.argmax(inner_cv_scores)
                bestincv_params = inner_cv_params[bestp_index]
                print('best internal cv params', bestincv_params)
                scalar = StandardScaler()
                X_train_c = pd.DataFrame(scalar.fit_transform(X_train_c))
                X_test = pd.DataFrame(scalar.transform(X_test))
                stacked = pd.concat([X_train_c, y_train_c], axis=1)
                cols = []
                cols.extend(range(X_train_c.shape[1]))  # the values zero to the number of columns
                cols.append(target_col.name)
                stacked.columns = cols
                if edge == 'fscore':
                    arr_outer = fscore(stacked, class_col=target_col.name)[:-1]
                    # fscore is different for the multiclass and binary case; has been incorporated above
                if edge == 'pearson':
                    arr_outer = stacked.corr().iloc[:, -1]
                arr_outer.fillna(0, inplace=True)

                arr_outer = np.array(arr_outer)
                val = np.nanpercentile(arr_outer, 100 - per)
                index = np.where(arr_outer >= val)

                X_train = X_train_c.iloc[:, index[0]]
                X_test = X_test.iloc[:, index[0]]
                if classifier == 'SVC':

                    clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_score = clf.predict_proba(X_test)

                if len(y_score[0]) == 2: #number of classes in the y_score
                    outer_cv_scores.append(compute_scores(y_test, y_pred, [x[1] for x in y_score], choice, metrics))
                else:
                    outer_cv_scores.append(compute_scores(y_test, y_pred, y_score, choice, metrics))
                outer_cv_params.append(bestincv_params)

            best_params[choice][per] = outer_cv_params[np.argmax(outer_cv_scores['balanced_accuracy'])]#need to see if this is in cv

            for metric in metrics:
                # validation set
                metric_score[choice][per][metric] = {}
                metric_score[choice][per][metric] = round(np.mean([scores[f'{metric}']for scores in outer_cv_scores]), 3)
                print(f'{metric}',np.mean([score[metric] for score in outer_cv_scores]))
                # out of bag error
    return {'Metrics': metric_score,'Parameters': best_params}


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
                for clf in combined.keys():
                    print("dictionary", combined[clf][choice])
                    validation.append(combined[clf][choice][top_per[k]][metrics[j]])

                ax[k][j].plot(list(combined.keys()), validation, marker='+', label=choice,
                              color=color, linestyle='dashed',markersize=12)
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

    for clf in ['SVC', 'RF', 'GB', 'MLP']:  # other ones are taking too long
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

