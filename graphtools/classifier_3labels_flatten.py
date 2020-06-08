import numpy as np
import pandas as pd
from train_classifier import train_SVC, train_RandomForest
from sklearn.model_selection import train_test_split
from readfiles import computed_subjects
from metrics import compute_scores
from processing import generate_combined_matrix, hist_fscore, hist_correlation
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform


# %%
def dict_classifier(whole, classifier, metrics, labels, big5, data, new_fscores):
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
    for i in range(5):  # different labels
        # print(labels[i], ':', big5[i])
        metric_score[big5[i]] = {}

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
                                 'gamma': loguniform(1e-4, 1e-3),
                                 'kernel': ['rbf'],
                                 'class_weight': ['balanced', None]}
                rcv = RandomizedSearchCV(clf, distributions, random_state=42, scoring=metrics, refit='roc_auc')
            elif classifier == 'RF':
                clf = RandomForestClassifier()
                distributions = {'bootstrap': [True, False],
                                 'max_depth': [10, 20, 30, 40, 50, 60, 70],
                                 'max_features': ['auto', 'sqrt'],
                                 'min_samples_leaf': [1, 2, 4],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]}
                rcv = RandomizedSearchCV(clf, distributions, random_state=42, scoring=metrics, refit='roc_auc')
            # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
            search = rcv.fit(X, Y)
            scores = search.cv_results_
            for metric in metrics:
                metric_score[big5[i]][per][metric] = np.mean(scores[f'mean_test_{metric}'])

    return metric_score


# %%
def make_csv(dict_score, filename):
    cv = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score.items()
    },
        axis=0)
    cv.to_csv(filename)


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
metrics = ['balanced accuracy', 'AUC', 'accuracy', 'F1 score']
cv_metrics = ['balanced_accuracy', 'roc_auc', 'accuracy', 'f1']
# %%
make_csv(dict_classifier(whole, 'RF', cv_metrics, labels, big5, data, new_fscores), 'outputs/RF_results_cv.csv')
make_csv(dict_classifier(whole, 'SVC', cv_metrics, labels, big5, data, new_fscores), 'outputs/SVM_results_cv.csv')
# %%
