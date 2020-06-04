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
def make_dictionary(option, classifier, metrics, labels, data, new_fscores):
    '''
    :param option: if we want the scores with or without cross validation
    :param classifier: the name of the classifier we want to test
    :param metrics: the name of the metrics we want to calculate
    :param labels: the big5 personality labels
    :param data: the file with contains the labels for all features of all subjects
    :param new_fscores: flattened array of f scores: num_subjects x num edges
    :return: metric_scores: the values to be calculated using permitted keywords
    '''
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

                if option == 'cv':
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.90, test_size=0.1,
                                                                        shuffle=True)
                    if classifier =='SVC':
                        y_pred, y_score = train_SVC(X_train, y_train, X_test)
                    elif classifier== 'RF':
                        y_pred, y_score = train_RandomForest(X_train, y_train, X_test)
                    scores = compute_scores(y_test, y_pred, y_score)

                if option == 'no':
                    if classifier == 'SVC':
                        clf = SVC(gamma='auto', probability=True)
                        scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                    elif classifier == 'RF':
                        rf = RandomForestClassifier(max_depth=10, random_state=10)
                        scores = cross_validate(rf, X, Y, cv=5, scoring=metrics)

                # according to the documentation the score must be for the higher class
                for metric, score in zip(metrics, scores):
                    metric_score[big5[i]][per][metric] = round(score, 3)
    return metric_score
#%%
def make_csv(dict_score, filename):
    cv = pd.concat({
            k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score.items()
        },
        axis=0)
    cv.to_csv(filename)

#%%
if __name__ == '__main__':
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
    cv_metrics = ['balanced_accuracy', 'roc_auc', 'accuracy', 'f1']
    make_csv(make_dictionary('no','RF', metrics, labels, data, new_fscores), 'RF_results.csv')
    make_csv(make_dictionary('no,'SVC', metrics, labels, data, new_fscores), 'SVM_results.csv')
    make_csv(make_dictionary('cv','RF', cv_metrics ,labels, data, new_fscores),'RF_results_cv.csv')
    make_csv(make_dictionary('cv','SVC', cv_metrics,labels, data, new_fscores),'SVM_results_cv.csv')

