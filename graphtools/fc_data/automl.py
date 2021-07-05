from load_data import *
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.metrics import log_loss, roc_auc_score, balanced_accuracy_score
import autosklearn.classification


dir = '/home/shreya/git/Thesis/graphtools/fc_data/data'
j = 1
f = open('fc_data/outputs/test_results.txt', 'w')
for groups in [['hc', 'ad'], ['hc', 'mci']]:
    print('Performance for groups', groups, file=f)
    combined_edges, combined_nodes = load_files(dir, groups=groups)
    # fscore_hist(combined_edges)  # fscores of the edges
    # fscore_hist(combined_nodes)  # fscores of the node features
    num_nodes = len(combined_nodes.columns) - 1  # because there is one column for the label here
    assert not sum(combined_edges.isna().any())
    fig, ax = plt.subplots(5, 1, figsize=(15, 10))
    skf = StratifiedKFold(n_splits=5)
    i = 0
    ax = ax.ravel()
    base_scores = []
    base_params = []
    indices = list(find_indices(shape=num_nodes).keys())
    X = combined_edges.iloc[:, indices]
    y = combined_edges.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=22)

    #automl = AutoML()
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)

    print(X_test.shape, predictions.shape, file=f)
    print("LogLoss", log_loss(y_test, predictions), file=f)
    print("AUC", roc_auc_score(y_test, predictions), file=f)
    print("Balanced accuracy", balanced_accuracy_score(y_test, predictions), file=f)
    j+=1

f.close()