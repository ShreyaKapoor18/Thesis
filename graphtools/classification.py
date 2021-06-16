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
from scipy.stats import ttest_ind

# %%

def feature_selection(X_train, X_val, y_train, per, feature):
    scalar2 = StandardScaler()
    print('feature selection')
    #assert len(np.unique(y_train)) > 2 #to make sure that we are getting the unbinned personality traits
    # print('Initial X_train, y_train, X_val, y_val',  X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    X_train = pd.DataFrame(scalar2.fit_transform(X_train), index=X_train.index)
    X_val = pd.DataFrame(scalar2.transform(X_val), index=X_val.index)

    stacked = pd.concat([X_train, y_train], axis=1)
    cols = []
    cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
    cols.append(y_train.name)
    stacked.columns = cols
    if feature == 'fscore':
        name = y_train.name
        stacked[name] = stacked[name] >= stacked[name].median()
        arr_inner = fscore(stacked, class_col=name)[:-1]
        # fscore is different for the multiclass and binary case; has been incorporated above
    if feature == 'pearson':
        arr_inner = stacked.corr().iloc[:-1, -1]
    if feature == 't_test':
        name = y_train.name
        stacked[name] = stacked[name] >= stacked[name].median()
        group0 = stacked[stacked[name] == 0]
        group1 = stacked[stacked[name] ==1]
        arr_inner = []
        for i in range(X_train.shape[1]):
            arr_inner.append(ttest_ind(group0.iloc[:, i], group1.iloc[:,i]).pvalue)

    arr_inner.fillna(0, inplace=True)
    arr_inner = np.array(arr_inner)
    val = np.nanpercentile(arr_inner, 100 - per)
    index = np.where(arr_inner >= val)

    X_train = X_train.iloc[:, index[0]]
    X_val = X_val.iloc[:, index[0]]

    assert list(X_train.index) == list(y_train.index)

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
def dict_classifier(classifier, X_train, X_test, y_train, y_test, metrics, feature, percent):
    """

    :param whole: the matrix containing the edge information for all subjects
    :param classifier: the name of the classifier we want to test
    :param metrics: the name of the metrics we want to calculate

    :return: metric_scores: the values to be calculated using permitted keywords
    """
    #try making a csv file for the same
    refit_metric = 'balanced_accuracy'
    print('The refit metric we are using', refit_metric)
    print('=' * 100)
    print('Classifier', classifier)
    metric_score = {}
    best_params = {}
    cols = ['Classifier', 'Parameter', 'Percentage', 'Choice', 'Refit Metric']
    cols.extend(metrics)
    results = []
    for per in percent:
        print('*' * 100)
        print('percentage of features being used:', per)
        metric_score[per] = {}
        best_params[per] = {}

        for choice in ['test throw median', 'keep median']:
            print('-'*100)
            assert len(X_train) == len(y_train)
            med = int(y_train.median())  # the median is tried based on the training set
            print('median of the training data', med)
            results.append([classifier, feature, per, choice, refit_metric])
            #print('value counts in train and test data set\n', y_train.value_counts(), y_test.value_counts())
            print('Descriptive statistics of the training and the test set labels', describe(y_train), describe(y_test))
            # all the things inside the loop, probably the percentage of features shall be inside the loop
            y_train_l = pd.qcut(y_train, 5, labels=False, retbins=True)[0]
            # we need to pass the non-binned values for effective pearson correlation calc.
            print('The number of subjects which are to be removed:', sum(y_train== 2))
            y_train_l = y_train_l[y_train != 2]
            y_train_l = y_train_l // 3#binarizing the values by removing the middle quartile
            X_train_l = X_train.loc[y_train_l.index]
            assert len(X_train_l) == len(y_train_l)
            print('The choice that we are using', choice)
            X_train_l, X_test_l = feature_selection(X_train_l, X_test, y_train_l, per, feature)
            if choice == 'test throw median':
                # removing subjects that are close to the median of the training data
                print(sum(abs(y_test - med) <= 1), 'The number of subjects with labels '
                                                   'within difference of 1.0 from the median value')
                #length_sub = sum(abs(y_test - med) <= 1)
                y_test_l = y_test[abs(y_test - med) > 1]  # maybe most of the values are close to the median
                y_test_l = y_test_l >=med  #binarizing the label check for duplicates

                X_test_l = X_test_l.loc[list(set(y_test_l.index))]
                # making sure that the training data is also for the same subjects
                assert len(X_test) == len(y_test)

            elif choice == 'keep median':
                y_test_l = y_test_l >= med# we just binarize it and don't do anything else
            # now we do the cross validation search
            metric_score[per][choice] = {}
            best_params[per][choice] = {}

            clf, distributions = get_distributions(classifier, True, None)
            rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                     refit=refit_metric, cv=5, n_iter=200,
                                     n_jobs=-1)
            # this is already producing 5 folds so we need to do something different?
            print('starting to train')
            search = rcv.fit(X_train_l, y_train_l)
            clf_out = search.best_estimator_
            plot_grid_search(search.cv_results_, refit_metric) #need to see what we will plot or not

            clf_out.fit(X_train_l, y_train_l)
            y_pred = clf_out.predict(X_test_l)
            assert len(y_pred) == len(X_test_l)
            outer_test = compute_scores(y_test_l, y_pred, clf_out.predict_proba(X_test_l)[:,1], metrics =metrics)
            ytrain_pred = clf_out.predict(X_train_l)
            outer_train = compute_scores(y_train_l, ytrain_pred, clf_out.predict_proba(X_train_l)[:,1], metrics=metrics)

            for metric in metrics:
                # validation set
                results[-1].append(outer_test[metric])
                metric_score[per][choice][metric] = {}
                metric_score[per][choice][metric]['test'] = round(outer_test[metric], 3)
                # take the mean of the outer validation scores for each of the different algorithms
                metric_score[per][choice][metric]['train'] = round(outer_train[metric], 3)
                best_params[per][choice][metric] = search.best_params_
                print(f'{metric} test score', round(outer_test[metric], 3))
                print(f'{metric} train score', round(outer_train[metric], 3))
            # out of bag error
    res = pd.DataFrame(results, columns=cols)
    res.to_csv('outputs/baseline.csv')
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
                              color=color, markersize=12)
                ax[k][j].plot(list(combined.keys()), train_score, marker='+', label=choice + 'train',
                              color=color, linestyle='dashed', markersize=12)
            if top_per[k] == 0:
                ax[k][j].set_title(f'100% features')
            else:
                ax[k][j].set_title(f'Top {top_per[k]}% features')
            ax[k][j].set_xlabel('Classifier')
            ax[k][j].set_ylabel(metrics[j])
            ax[k][j].grid(which='minor', alpha=0.2)
            ax[k][j].grid(which='major', alpha=0.5)
    plt.ylim(0.4, 1)
    plt.legend(legends)
    fig.suptitle(target)
    plt.tight_layout()
    plt.savefig(f'outputs/figures/classification_{target}')
    plt.show()


# %%
def run_classification(X_train, X_test,y_train, y_test, metrics, target, feature):
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
    classifiers = ['RF', 'MLP', 'SVC']
    for clf in classifiers:  # other ones are taking too long
        start = time.time()
        d1 = dict_classifier(clf, X_train,X_test, y_train, y_test, metrics, feature, [5,10,50,100])
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
