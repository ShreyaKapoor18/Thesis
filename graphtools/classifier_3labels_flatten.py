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
            val = np.percentile(new_fscores[i], 100 - per)
            index = np.where(new_fscores[i] >= val)
            # print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))
            Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
            X = whole[:, index[0]]

            if classifier == 'SVC':
                clf = SVC()
                distributions = {'C': loguniform(1e0, 1e3),
                                 'gamma': loguniform(1e-4, 1e-2),
                                 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                                 'class_weight': ['balanced', None]}
                print(f'Executing {clf}')
            elif classifier == 'RF':
                clf = RandomForestClassifier()
                distributions = {'bootstrap': [True, False],
                                 'max_depth': [10, 20, 30, 40, 50, 60, 70],
                                 'max_features': ['auto', 'sqrt'],
                                 'min_samples_leaf': [1, 2, 4],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]}
                print(f'Executing {clf}')
            elif classifier == 'GB':
                clf = GradientBoostingClassifier()
                distributions = {'loss': ['deviance', 'exponential'],
                                 'learning_rate': [0.8, 0.9, 1],
                                 'tol': loguniform(1e-4, 1e-2),
                                 'min_samples_leaf': [1, 2, 4],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]
                                 }
                print(f'Executing {clf}')
            elif classifier == 'MLP':
                clf = MLPClassifier()
                distributions = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                                 'activation': ['tanh', 'relu'],
                                 'solver': ['sgd', 'adam'],
                                 'alpha': [0.0001, 0.05],
                                 'learning_rate': ['constant', 'adaptive']}
                print(f'Executing {clf}')

            rcv = RandomizedSearchCV(clf, distributions, random_state=42, scoring=metrics, refit='roc_auc')
            # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
            search = rcv.fit(X, Y)
            scores = search.cv_results_
            best_params[big5[i]][per] = search.best_params_
            for metric in metrics:
                metric_score[big5[i]][per][metric] = round(np.mean(scores[f'mean_test_{metric}']), 3)

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
def visualise_performance(combined, *args):
    # for each label we will visualise the performance of different classifiers
    for i, label in zip(range(5), big5):
        fig, ax = plt.subplots(5, len(metrics))
        for j, metric in zip(range(len(metrics)), metrics):
            l = []
        for clf in combined.keys():
            l.append(combined[clf][big5[i]][metric])
        ax[i][j].plot(len(combined.keys()), l)
        ax[i][j].set_xticks(clf)
        ax[i][j].set_xlabel('Classifier')
        ax[i][j].set_ylabel(metric)
        plt.savefig(f'outputs/classification_{label}')
# %%
data = computed_subjects()  # labels for the computed subjects
data.reset_index(inplace=True)
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri)
# The labels i.e. the ones from unrestricted_files!
# %%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
'Extraversion']
# %%
fscores = hist_fscore(data, whole, labels, big5, edge_names, tri)
# %%
# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1] * fscores.shape[2]))
metrics = ['balanced_accuracy', 'roc_auc', 'accuracy', 'f1']
# %%
combined = {}
for clf in ['SVC', 'RF', 'GB', 'MLP']:
    d1 = dict_classifier(clf, whole, metrics, labels, big5, data, new_fscores)
make_csv(d1, f'outputs/{clf}_results_cv.csv')
combined[clf] = d1['Metrics']
# %%
visualise_performance(combined, big5)
