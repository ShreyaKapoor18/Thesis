import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.neural_network import MLPClassifier
from processing import generate_combined_matrix, hist_fscore
from readfiles import computed_subjects
import matplotlib.pyplot as plt
import time
import datetime
import json

#%%

def data_splitting(choice, i, index, *args, **kwargs):
    if choice == 'qcut':
        # choice to cut into three quartiles
        y = np.array(pd.qcut(data[labels[i]], 3, labels=False, retbins=True)[0])
        X = whole[:, index[0]]
    if choice == 'median':
        # choice to threshold around the median
        y = np.array(data[labels[i]]>=data[labels[i]].median())
        X = whole[:, index[0]]
    if choice == 'throw median':
        y = pd.qcut(data[labels[i]], 5, labels=False, retbins=True)[0]
        y.reset_index(drop=True, inplace=True)
        print(sum(y==2), 'is the number of subjects which have been removed')
        y = y[y!=2]
        y = y//3 # 0 and 1 classes get mapped to 0 and 3,4 get mapped to 1
        print(len(y), 'New number of subjects in our dataset')
        #X = whole[y.index, index[0]] don't know why this type of slicin is not working
        X = np.array([whole[i, index[0]] for i in y.index])

    return X, y

# %%
def dict_classifier(classifier, *args):
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

        for per in [5, 10, 50, 0]:
            metric_score[big5[i]][per] = {}
            best_params[big5[i]][per]= {}
            val = np.percentile(new_fscores[i], 100 - per)
            index = np.where(new_fscores[i] >= val)
            # print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))
            # Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
            for choice in ['qcut', 'median', 'throw median']:
                metric_score[big5[i]][per][choice] = {}
                X,y = data_splitting(choice, i, index, data, whole)
                if classifier == 'SVC':
                    clf = SVC(probability=True)
                    distributions = {'C': loguniform(1e0, 1e3),
                                     'gamma': loguniform(1e-4, 1e-2),
                                     'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                                     'class_weight': ['balanced', None]}

                elif classifier == 'RF':
                    clf = RandomForestClassifier()
                    distributions = {'bootstrap': [True, False],
                                     'max_depth': [10, 20, 30, 40, 50, 60, 70],
                                     'max_features': ['auto', 'sqrt'],
                                     'min_samples_leaf': [1, 2, 4],
                                     'min_samples_split': [2, 5, 10],
                                     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]}

                elif classifier == 'GB':
                    clf = GradientBoostingClassifier()
                    distributions = {  # 'loss': ['deviance', 'exponential']
                        'learning_rate': [0.8, 0.9, 1],
                        'tol': [0.01, 0.1],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        'n_estimators': [200, 400] #takes too long to converge if tolerance not specificied
                    }
                    # multiclass cannot use losss exponential
                elif classifier == 'MLP':
                    clf = MLPClassifier()
                    distributions = {'hidden_layer_sizes': [(50, 100, 100, 50), (50, 100, 50)],
                                     'activation': ['tanh', 'relu'],
                                     'solver': ['sgd', 'adam'],
                                     'alpha': [0.001, 0.05],
                                     'learning_rate': ['adaptive']} #doesn't converge even with maximum iterations!

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
    # for each label we will visualise the performance of different classifiers
    for i in range(len(big5)):
        fig, ax = plt.subplots(len(top_per), len(metrics), figsize=(25, 20))
        for k in range(len(top_per)):
            for j in range(len(metrics)):

                for choice in ['qcut', 'median', 'throw median']:
                    l = []
                    for clf in combined.keys():
                        l.append(combined[clf][big5[i]][top_per[k]][choice][metrics[j]])
                        # print(clf, big5[i], top_per[k], metrics[j])
                        # print(combined[clf][big5[i]][top_per[k]][metrics[j]])
                    ax[k][j].scatter(combined.keys(), l, label=choice)
                    ax[k][j].plot(list(combined.keys()), l)
                    # print('xx', len(l))
                ax[k][j].legend(loc='lower right')
                # ax[k][j].set_xticks(list(combined.keys()))
                if top_per[k] == 0:
                    ax[k][j].set_title(f'100% features')
                else:
                    ax[k][j].set_title(f'Top {top_per[k]}% features')
                ax[k][j].set_xlabel('Classifier')
                ax[k][j].set_ylabel(metrics[j])
        fig.suptitle(big5[i])
        plt.tight_layout()
        plt.savefig(f'outputs/classification_{big5[i]}')
        # plt.show()


# %%
if __name__ == "__main__":

    data = computed_subjects()  # labels for the computed subjects
    num = 84  # number of nodes in the graph
    tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
    whole, order = generate_combined_matrix(tri, list(data.index))
    # The labels i.e. the ones from unrestricted_files!

    labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
    edge_names = ['mean_FA', 'mean strl', 'num streamlines']
    big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
            'Extraversion']

    fscores = hist_fscore(data, whole, labels, big5, edge_names, tri)

    # without taking the edge type into consideration
    new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1] * fscores.shape[2]))
    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

    combined = {}
    for clf in ['SVC', 'RF', 'GB', 'MLP'][:2]: #other ones are taking too long
        start = time.time()
        d1 = dict_classifier(clf, whole, metrics, labels, big5, data, new_fscores)
        end = time.time()
        print(f'Time taken for {clf}: {datetime.timedelta(seconds=end-start)}')
        make_csv(d1, f'outputs/{clf}_results_cv.csv')
        combined[clf] = d1['Metrics']
    with open('combined_dict.txt', 'w') as f:
        f.write(json.dumps(combined)) # write the combined dictionary to the file so that this can be read later on
    visualise_performance(combined, big5, metrics, [5, 10, 50, 0])
