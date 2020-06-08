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
#%%
def dict_classifier(whole, classifier, metrics, labels,big5, data, new_fscores):
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
    for i in range(5):# different labels
        #print(labels[i], ':', big5[i])
        metric_score[big5[i]] = {}

        for per in [5, 10, 15, 20]:
                metric_score[big5[i]][per] = {}
                val = np.percentile(new_fscores[i], 100-per)
                index = np.where(new_fscores[i] >= val)
                #print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))
                Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
                X = whole[:, index[0]]

                if classifier == 'SVC':
                    clf = SVC(gamma='auto', probability=True)
                elif classifier == 'RF':
                    clf = RandomForestClassifier(max_depth=10, random_state=10)
                scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                for metric in metrics:
                    metric_score[big5[i]][per][metric] = scores[f'test_{metric}']

    return metric_score
#%%
def make_csv(dict_score, filename):
    cv = pd.concat({
            k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score.items()
        },
        axis=0)
    cv.to_csv(filename)

#%%
data = computed_subjects() # labels for the computed subjects
data.reset_index(inplace= True)
num = 84 # number of nodes in the graph
tri = int(num * (num + 1) * 0.5) #we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri)
# The labels i.e. the ones from unrestricted_files!
#%%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E' ]
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                        'Extraversion']
#%%
fscores = hist_fscore(data, whole, labels,big5, edge_names, tri)
#%%
# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1]*fscores.shape[2]))
metrics = ['balanced accuracy', 'AUC', 'accuracy', 'F1 score']
cv_metrics = ['balanced_accuracy', 'roc_auc', 'accuracy', 'f1']
#%%
make_csv(dict_classifier(whole,  'RF', cv_metrics, labels, big5, data, new_fscores), 'outputs/RF_results_cv.csv')
make_csv(dict_classifier(whole, 'SVC', cv_metrics, labels, big5, data, new_fscores), 'outputs/SVM_results_cv.csv')
#%%

