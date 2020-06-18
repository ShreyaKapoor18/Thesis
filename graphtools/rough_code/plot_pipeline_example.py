"""
====================================================
Example of building a graph classification pipeline.
====================================================

Script makes use of :class:`grakel.ShortestPath`
"""
from __future__ import print_function
print(__doc__)

import numpy as np
from processing import generate_combined_matrix, hist_fscore
from readfiles import computed_subjects
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset, get_dataset_info

from grakel.kernels import ShortestPath
from grakel import Graph
#%%
# Loads the Mutag dataset from:
MUTAG = fetch_dataset("MUTAG", verbose=False)
#get_dataset_info('MUTAG')
G, y = MUTAG.data, MUTAG.target

#%%
data = computed_subjects()  # labels for the computed subjects
data.reset_index(inplace=True)
y = np.array(data['NEOFAC_A']>=data['NEOFAC_A'].median()).astype(int)

num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri)
# Values of C parameter of SVM
wholex = np.reshape(whole,(whole.shape[0], whole.shape[1]//3, 3))
mat = np.triu_indices(84)
G = []
#nodes = {x: 1 for x in range(84)}
for i in range(len(wholex)):
    edges = {(0,0)}
    #edges = {}
    edge_attributes = {}
    for j in range(len(mat[0])):
        edges.add((mat[0][j], mat[1][j])) # first feature is mean fa between nodes
        #edges[(mat[0][j], mat[1][j])] = 1
        #edge_attributes[(mat[0][j], mat[1][j])] = wholex[i,j,:]
    #print (edges, edge_attributes)
    G.append(Graph(edges)) #edge_labels=edge_attributes))

    #G.append([edges, nodes, edge_attributes])
#%%

C_grid = (10. ** np.arange(-4,6,1) / len(G)).tolist()

# Creates pipeline
estimator = make_pipeline(
    ShortestPath(normalize=True, with_labels=False),
    GridSearchCV(SVC(kernel='precomputed'), dict(C=C_grid),
                 scoring='accuracy', cv=10))

# Performs cross-validation and computes accuracy
n_folds = 10
acc = accuracy_score(y, cross_val_predict(estimator, G, y, cv=n_folds))
print("Accuracy:", str(round(acc*100, 2)) + "%")
#%%
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)

# Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")
