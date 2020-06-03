import numpy as np
import pandas as pd
from train_classifier import train_SVC, train_RandomForest
from sklearn.model_selection import train_test_split
from readfiles import computed_subjects
from metrics import compute_scores
from processing import generate_combined_matrix, hist_fscore, hist_correlation
#%%
def make_dictionary(classifier, metrics, labels, data, new_fscores):
    metric_score = {}
    for i in range(5):# different labels
        print(labels[i], ':', big5[i])
        metric_score[big5[i]] = {}

        for per in [5, 10, 15, 20]:
                metric_score[big5[i]][per] = {}
                val = np.percentile(new_fscores[i], 100-per)
                index = np.where(new_fscores[i] >= val)
                #print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))
                Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
                X = whole[:, index[0]]

                X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.90, test_size=0.1, shuffle=True)

                if classifier =='SVC':
                    y_pred, y_score = train_SVC(X_train, y_train, X_test)
                elif classifier== 'RF':
                    y_pred, y_score = train_RandomForest(X_train, y_train, X_test)
                # according to the documentation the score must be for the higher class
                scores = compute_scores(y_test, y_pred, y_score)
                for metric, score in zip(metrics, scores):
                    metric_score[big5[i]][per][metric] = round(score, 3)
    return metric_score
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
hist_correlation(data, whole, labels, edge_names, big5, tri)
#%%
# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1]*fscores.shape[2]))
metrics = ['balanced accuracy', 'AUC', 'accuracy', 'F1 score']
RF_score = make_dictionary('RF', metrics, labels, data, new_fscores)
SVM_score = make_dictionary('SVC', metrics, labels, data, new_fscores)
#%%
RF = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in RF_score.items()
    },
    axis=0)
SVM = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in SVM_score.items()
    },
    axis=0)
#%%
RF.to_csv('RF_results.csv')
SVM.to_csv('SVM_results.csv')